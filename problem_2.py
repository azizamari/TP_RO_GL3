from gurobipy import *
import pandas as pd


def solve(
    price_min=0.10,
    price_max=0.50,
    capacity_multiplier=1.0,
    demand_sensitivity=1.0,
    # Extensions souhaitées
    segments=None,                 # list of segment names, e.g. ['res','comm','ind']
    segment_shares=None,           # relative share of base demand per segment
    battery=None,                  # dict with battery params or None
    max_seg_price_diff=0.20,       # regulatory fairness: max spread between segments per period
    equity_penalty=0.0,            # linear penalty coeff (€ per € of spread) for multi-objective
    verbose=True,
    return_dict=False
):
    """
    Résout le problème de tarification optimale de l'électricité.
    
    Args:
        price_min (float): Prix minimum autorisé (€/kWh)
        price_max (float): Prix maximum autorisé (€/kWh)
        capacity_multiplier (float): Multiplicateur de capacité (1.0 = 100%)
        demand_sensitivity (float): Sensibilité de la demande au prix (élasticité)
        verbose (bool): Afficher les résultats détaillés
        return_dict (bool): Retourner un dictionnaire (pour Gradio)
    
    Returns:
        Model ou dict: Modèle Gurobi résolu ou dictionnaire de résultats
    """
    
    if verbose:
        print("\n" + "=" * 80)
        print("PROBLÈME 2 - TARIFICATION OPTIMALE DE L'ÉLECTRICITÉ")
        print("=" * 80)
    
    try:
        # ====================================================================
        # DONNÉES DU PROBLÈME
        # ====================================================================
        
        # Périodes de la journée (24 heures divisées en 6 périodes)
        periodes = [
            "Nuit (0h-4h)",
            "Matin tôt (4h-8h)",
            "Matin (8h-12h)",
            "Après-midi (12h-16h)",
            "Soirée (16h-20h)",
            "Nuit tardive (20h-24h)"
        ]
        n_periodes = len(periodes)
        
        # Demande de base (kWh) - demande maximale si prix = 0
        # Plus élevée pendant les heures de pointe
        demande_base = [
            5000,   # Nuit - faible demande
            8000,   # Matin tôt - demande croissante
            12000,  # Matin - haute demande (entreprises)
            10000,  # Après-midi - demande modérée
            15000,  # Soirée - POINTE (retour maisons, cuisine, etc.)
            7000    # Nuit tardive - demande décroissante
        ]
        
        # Élasticité de la demande (sensibilité au prix)
        # Plus la valeur est élevée, plus la demande réagit au prix
        elasticite_base = [
            8000,   # Nuit - faible élasticité
            12000,  # Matin tôt
            20000,  # Matin - haute élasticité
            15000,  # Après-midi
            25000,  # Soirée - très élastique (consommateurs sensibles)
            10000   # Nuit tardive
        ]
        
        # Appliquer le facteur de sensibilité de la demande
        elasticite = [e * demand_sensitivity for e in elasticite_base]
        
        # Capacité de production par période (kWh)
        capacite_base = [
            6000,   # Nuit
            9000,   # Matin tôt
            13000,  # Matin
            11000,  # Après-midi
            16000,  # Soirée
            8000    # Nuit tardive
        ]
        
        # Appliquer le multiplicateur de capacité
        capacite = [c * capacity_multiplier for c in capacite_base]
        
        # Coût de production par kWh par période (€/kWh)
        # Plus élevé pendant les heures de pointe (centrales d'appoint)
        cout_production = [
            0.05,   # Nuit - coût faible (centrale de base)
            0.08,   # Matin tôt
            0.12,   # Matin - coût plus élevé
            0.10,   # Après-midi
            0.15,   # Soirée - coût maximal (centrales d'appoint)
            0.07    # Nuit tardive
        ]
        
        # Variation maximale de prix entre périodes consécutives (€/kWh)
        # Pour éviter des chocs de prix trop brutaux
        delta_prix_max = 0.15

        # ------------------------------------------------------------------
        # Paramètres d'extensions: segmentation et batterie
        # ------------------------------------------------------------------
        if segments is None:
            segments = ['res', 'comm', 'ind']
        n_segments = len(segments)

        # par défaut, répartir la demande de base entre segments (résidentiel majoritaire)
        if segment_shares is None:
            segment_shares = [0.6, 0.25, 0.15]
        # normaliser
        total_shares = sum(segment_shares)
        segment_shares = [s / total_shares for s in segment_shares]

        # construire a[t,s] et b[t,s]
        demande_base_seg = [[demande_base[t] * segment_shares[s] for s in range(n_segments)] for t in range(n_periodes)]
        # ajuster élasticité par segment (ex: industries moins élastiques)
        segment_elasticity_factor = [1.0, 1.2, 0.8][:n_segments]
        elasticite_seg = [[elasticite[t] * segment_elasticity_factor[s] for s in range(n_segments)] for t in range(n_periodes)]

        # Batterie: paramétrage simple
        # battery = {
        #   'E_max': 20000, 'P_charge_max': 3000, 'P_discharge_max':3000,
        #   'eta_c':0.95, 'eta_d':0.95, 'soc_init':0
        # }
        if battery is None:
            battery = {
                'E_max': 20000.0,
                'P_charge_max': 3000.0,
                'P_discharge_max': 3000.0,
                'eta_c': 0.95,
                'eta_d': 0.95,
                'soc_init': 0.0
            }

        # ------------------------------------------------------------------
        # Validation rapide des paramètres pour éviter infaisabilité triviale
        # Vérifie que pour au moins un prix dans [price_min, price_max], la demande
        # calculée a - b*p peut être >= 0. Si a - b*price_min < 0 -> impossible.
        problematic = []
        for t in range(n_periodes):
            for s in range(n_segments):
                a_ts = demande_base_seg[t][s]
                b_ts = elasticite_seg[t][s]
                # demande au prix minimal (meilleure chance d'avoir demande positive)
                q_at_price_min = a_ts - b_ts * price_min
                if q_at_price_min < 0:
                    problematic.append((t, s, a_ts, b_ts, q_at_price_min))

        if problematic:
            msg_lines = [
                "Paramètres incompatibles détectés: certaines demandes segmentées sont négatives même au prix minimum."
            ]
            for (t, s, a_ts, b_ts, q_min) in problematic:
                msg_lines.append(f"  - Période {t} ('{periodes[t]}'), segment '{segments[s]}': a={a_ts}, b={b_ts}, a - b*p_min = {q_min:.2f}")
            msg_lines.append("Suggestion: diminuer 'price_min' ou ajuster 'segment_shares'/'demande_base' pour ces périodes.")
            msg = "\n".join(msg_lines)
            if verbose:
                print("\nERROR: paramètres initiaux invalides pour la demande segmentée:")
                print(msg)
            if return_dict:
                return {'status': 'infeasible_params', 'error': msg}
            raise ValueError(msg)
        
        # ====================================================================
        # CRÉATION DU MODÈLE
        # ====================================================================
        
        model = Model("Tarification_Electricite")
        
        # ====================================================================
        # VARIABLES DE DÉCISION (EXTENDUES: segments + batterie + fairness)
        # ====================================================================

        # Prix par période et par segment
        p = model.addVars(range(n_periodes), range(n_segments), vtype=GRB.CONTINUOUS,
                          name='prix', lb=price_min, ub=price_max)

        # Quantités par période et par segment
        q = model.addVars(range(n_periodes), range(n_segments), vtype=GRB.CONTINUOUS,
                          name='quantite', lb=0)

        # Variables batterie: charge, discharge et état de charge (SoC)
        charge = model.addVars(range(n_periodes), vtype=GRB.CONTINUOUS,
                               name='charge', lb=0, ub=battery['P_charge_max'])
        discharge = model.addVars(range(n_periodes), vtype=GRB.CONTINUOUS,
                                  name='discharge', lb=0, ub=battery['P_discharge_max'])
        soc = model.addVars(range(n_periodes), vtype=GRB.CONTINUOUS,
                            name='soc', lb=0, ub=battery['E_max'])

        # Moyenne de prix par période (pour continuité) et extrêmes pour fairness
        p_avg = model.addVars(range(n_periodes), vtype=GRB.CONTINUOUS,
                              name='pavg', lb=price_min, ub=price_max)
        p_max = model.addVars(range(n_periodes), vtype=GRB.CONTINUOUS,
                              name='pmax', lb=price_min, ub=price_max)
        p_min = model.addVars(range(n_periodes), vtype=GRB.CONTINUOUS,
                              name='pmin', lb=price_min, ub=price_max)

        # ====================================================================
        # FONCTION OBJECTIF: MAXIMISER LE PROFIT MOINS PÉNALITÉ D'ÉQUITÉ
        # profit = Σ_t Σ_s (p[t,s]*q[t,s] - c[t]*q[t,s])
        # penalty = equity_penalty * Σ_t (p_max[t] - p_min[t])
        # ====================================================================

        profit_expr = quicksum((p[t, s] * q[t, s] - cout_production[t] * q[t, s])
                               for t in range(n_periodes) for s in range(n_segments))
        spread_expr = quicksum((p_max[t] - p_min[t]) for t in range(n_periodes))

        model.setObjective(profit_expr - equity_penalty * spread_expr, GRB.MAXIMIZE)

        # ====================================================================
        # CONTRAINTES
        # ====================================================================

        # 1. Demande élastique par segment: q[t,s] = a[t,s] - b[t,s]*p[t,s]
        for t in range(n_periodes):
            for s in range(n_segments):
                model.addConstr(q[t, s] == demande_base_seg[t][s] - elasticite_seg[t][s] * p[t, s],
                                name=f"Demande_Elastique_t{t}_s{s}")

        # 2. Capacité de production ajustée par charge/discharge batterie
        for t in range(n_periodes):
            model.addConstr(quicksum(q[t, s] for s in range(n_segments)) + charge[t]
                            <= capacite[t] + discharge[t], name=f"Capacite_{t}")

        # 3. Continuité des prix sur la moyenne des segments
        for t in range(n_periodes):
            model.addConstr(p_avg[t] == (1.0 / n_segments) * quicksum(p[t, s] for s in range(n_segments)),
                            name=f"PAvgDef_{t}")

        for t in range(n_periodes - 1):
            model.addConstr(p_avg[t+1] - p_avg[t] <= delta_prix_max, name=f"Var_Prix_Max_{t}")
            model.addConstr(p_avg[t] - p_avg[t+1] <= delta_prix_max, name=f"Var_Prix_Min_{t}")

        # Cycle 24h sur p_avg
        model.addConstr(p_avg[0] - p_avg[n_periodes-1] <= delta_prix_max, name="Cycle_PMax")
        model.addConstr(p_avg[n_periodes-1] - p_avg[0] <= delta_prix_max, name="Cycle_PMin")

        # 4. Fairness: p_max and p_min definitions and max spread constraint
        for t in range(n_periodes):
            for s in range(n_segments):
                model.addConstr(p_max[t] >= p[t, s], name=f"Pmax_def_t{t}_s{s}")
                model.addConstr(p_min[t] <= p[t, s], name=f"Pmin_def_t{t}_s{s}")
            model.addConstr(p_max[t] - p_min[t] <= max_seg_price_diff, name=f"MaxSpread_t{t}")

        # 5. Batterie: SoC dynamique
        eta_c = battery['eta_c']
        eta_d = battery['eta_d']
        for t in range(n_periodes):
            if t == 0:
                model.addConstr(soc[0] == battery['soc_init'] + eta_c * charge[0] - discharge[0] / eta_d,
                                name='SoC_0')
            else:
                model.addConstr(soc[t] == soc[t-1] + eta_c * charge[t] - discharge[t] / eta_d,
                                name=f'SoC_{t}')

        
        # ====================================================================
        # RÉSOLUTION
        # ====================================================================
        
        model.setParam('OutputFlag', 1 if verbose else 0)
        model.setParam('TimeLimit', 300)
        model.optimize()
        
        # ====================================================================
        # TRAITEMENT DES RÉSULTATS
        # ====================================================================
        
        if model.status == GRB.OPTIMAL:

            # Calcul des métriques étendues
            profit_total = model.objVal
            revenu_total = sum(p[t, s].x * q[t, s].x for t in range(n_periodes) for s in range(n_segments))
            cout_total = sum(cout_production[t] * sum(q[t, s].x for s in range(n_segments)) for t in range(n_periodes))
            quantite_totale = sum(q[t, s].x for t in range(n_periodes) for s in range(n_segments))
            prix_moyen = (revenu_total / quantite_totale) if quantite_totale > 0 else 0

            if verbose:
                print("\n" + "=" * 80)
                print("✓ SOLUTION OPTIMALE TROUVÉE (EXTENDUE)")
                print("=" * 80)

                print(f"\n📊 RÉSULTATS GLOBAUX:")
                print(f"  Profit total: {profit_total:,.2f} €")
                print(f"  Revenu total: {revenu_total:,.2f} €")
                print(f"  Coût total: {cout_total:,.2f} €")
                print(f"  Quantité totale vendue: {quantite_totale:,.0f} kWh")
                print(f"  Prix moyen pondéré: {prix_moyen:.3f} €/kWh")

                print(f"\n⚡ TARIFICATION PAR PÉRIODE (avec segments):")
                header = f"{'Période':<25} {'Prix_moy':<10} {'Quantité':<12}"
                for s in range(n_segments):
                    header += f" {segments[s]+'_prix':<12} {segments[s]+'_q':<12}"
                print(header)
                print("-" * 120)

                for t in range(n_periodes):
                    q_total_t = sum(q[t, s].x for s in range(n_segments))
                    prix_moy_t = sum(p[t, s].x for s in range(n_segments)) / n_segments
                    row = f"{periodes[t]:<25} {prix_moy_t:>8.3f} {q_total_t:>12.0f}"
                    for s in range(n_segments):
                        row += f" {p[t, s].x:>10.3f} {q[t, s].x:>12.0f}"
                    print(row)

            # Pour Gradio / sortie dict
            if return_dict:
                results_data = []
                for t in range(n_periodes):
                    q_total_t = sum(q[t, s].x for s in range(n_segments))
                    revenu_t = sum(p[t, s].x * q[t, s].x for s in range(n_segments))
                    cout_t = sum(cout_production[t] * q[t, s].x for s in range(n_segments))
                    profit_t = revenu_t - cout_t
                    util_t = (q_total_t / capacite[t]) * 100 if capacite[t] > 0 else 0

                    row = {
                        'Période': periodes[t],
                        'Prix_moyen (€/kWh)': round(sum(p[t, s].x for s in range(n_segments)) / n_segments, 3),
                        'Quantité totale (kWh)': round(q_total_t, 0),
                        'Revenu (€)': round(revenu_t, 2),
                        'Coût (€)': round(cout_t, 2),
                        'Profit (€)': round(profit_t, 2),
                        'Utilisation (%)': round(util_t, 1)
                    }
                    for s in range(n_segments):
                        row[f'Prix_{segments[s]} (€/kWh)'] = round(p[t, s].x, 3)
                        row[f'Quant_{segments[s]} (kWh)'] = round(q[t, s].x, 0)
                    results_data.append(row)

                # battery summary
                battery_summary = {
                    'soc_final': soc[n_periodes-1].x if n_periodes > 0 else None,
                    'total_charged': sum(charge[t].x for t in range(n_periodes)),
                    'total_discharged': sum(discharge[t].x for t in range(n_periodes))
                }

                return {
                    'status': 'optimal',
                    'objective': profit_total,
                    'revenu_total': revenu_total,
                    'cout_total': cout_total,
                    'quantite_totale': quantite_totale,
                    'prix_moyen': prix_moyen,
                    'marge': (profit_total / revenu_total) * 100 if revenu_total > 0 else 0,
                    'results': results_data,
                    'battery': battery_summary,
                    'model': model
                }
        
        elif model.status == GRB.INFEASIBLE:
            if verbose:
                print("\n" + "=" * 80)
                print("✗ PROBLÈME INFAISABLE")
                print("=" * 80)
                print("\nLe problème n'a aucune solution réalisable.")
                print("Causes possibles:")
                print("  - Prix min/max incompatibles avec la demande")
                print("  - Capacité de production insuffisante")
                print("  - Contraintes de variation de prix trop strictes")
            
            # Compute IIS and write file for diagnostics
            try:
                model.computeIIS()
                iis_file = "problem_2_iis.ilp"
                model.write(iis_file)
                if verbose:
                    print(f"IIS écrit dans: {iis_file}")
            except Exception:
                iis_file = None

            if return_dict:
                return {
                    'status': 'infeasible',
                    'objective': None,
                    'error': 'Problème infaisable - vérifiez les contraintes',
                    'iis': iis_file
                }
        
        elif model.status == GRB.UNBOUNDED:
            if verbose:
                print("\n" + "=" * 80)
                print("⚠ PROBLÈME NON BORNÉ")
                print("=" * 80)
            
            if return_dict:
                return {
                    'status': 'unbounded',
                    'objective': None,
                    'error': 'Problème non borné'
                }
        
        else:
            if verbose:
                print(f"\nStatut d'optimisation: {model.status}")
            
            if return_dict:
                return {
                    'status': f'status_{model.status}',
                    'objective': None
                }
        
        if verbose:
            print("=" * 80)
        
        # Exporter le modèle
        model.write("problem_2_tarification.lp")

        # Retour final — si on ne demande pas de dict, renvoyer le modèle
        if return_dict:
            # Si on arrive ici sans être passé par la section 'optimal', renvoyer un résumé minimal
            return {
                'status': f'status_{model.status}',
                'objective': model.objVal if hasattr(model, 'objVal') else None
            }
        return model
    
    except GurobiError as e:
        error_msg = f"Erreur Gurobi: {e}"
        if verbose:
            print(f"\n❌ {error_msg}")
        if return_dict:
            return {'status': 'error', 'error': error_msg}
        raise
    
    except Exception as e:
        error_msg = f"Erreur inattendue: {e}"
        if verbose:
            print(f"\n❌ {error_msg}")
            import traceback
            traceback.print_exc()
        if return_dict:
            return {'status': 'error', 'error': error_msg}
        raise


def get_problem_info():
    """
    Retourne les informations sur ce problème.
    
    Returns:
        dict: Métadonnées du problème
    """
    return {
        'name': 'Problème 2 - Tarification Optimale de l\'Électricité',
        'type': 'QP / MILP hybride (segmentation, batterie, contraintes de fairness)',
        'description': 'Optimisation des prix par segment, intégrant stockage et contraintes réglementaires',
        'author': '[Votre Nom]',
        'variables': 'prix et quantités par période et segment, variables batterie (charge/discharge/SoC), auxiliaires',
        'constraints': 'demande élastique par segment, capacité, continuité, fairness, SoC dynamique',
        'objective': 'Maximiser le profit - pénalités d\'inégalité (optionnel)'
    }


# =========================================================================
# TESTS ET EXÉCUTION
# =========================================================================
if __name__ == "__main__":
    print("🧪 Test du problème de tarification optimale\n")
    
    # Test 1: Configuration par défaut
    print("=" * 80)
    print("TEST 1: Configuration standard")
    print("=" * 80)
    model = solve(
        price_min=0.10,
        price_max=0.50,
        capacity_multiplier=1.0,
        demand_sensitivity=1.0,
        verbose=True
    )
    
    # Test 2: Capacité réduite
    print("\n\n" + "=" * 80)
    print("TEST 2: Capacité réduite (80%)")
    print("=" * 80)
    model2 = solve(
        price_min=0.10,
        price_max=0.50,
        capacity_multiplier=0.8,
        demand_sensitivity=1.0,
        verbose=True
    )
    
    # Test 3: Demande plus sensible au prix
    print("\n\n" + "=" * 80)
    print("TEST 3: Demande très élastique (2x)")
    print("=" * 80)
    model3 = solve(
        price_min=0.10,
        price_max=0.50,
        capacity_multiplier=1.0,
        demand_sensitivity=2.0,
        verbose=True
    )
    
    print("\n✓ Tous les tests terminés!")