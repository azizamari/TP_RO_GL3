import gradio as gr
import pandas as pd
from gurobipy import *
from problem_11_4_routage_du_personnel import *
import io
import sys

import problem_2 as problem_17_2
from problem_location_allocation import LocationAllocationSolver
import matplotlib.pyplot as plt


def solve_problem_9_4(budget_y1, budget_y2, budget_y3):
    output = io.StringIO()
    sys.stdout = output
    
    try:
        projets = [
            "Centrale Thermique Nord",
            "Centrale Solaire Sud",
            "Centrale Éolienne Offshore",
            "Centrale Hydroélectrique Est",
            "Centrale Nucléaire Centre",
            "Parc Éolien Terrestre Ouest",
            "Centrale Biomasse Rurale",
            "Centrale Géothermique"
        ]
        
        n_projets = len(projets)
        van = [45, 38, 62, 55, 120, 35, 28, 42]
        n_periodes = 3
        periodes = range(n_periodes)
        couts = [
            [15, 20, 10],
            [12, 15, 11],
            [25, 20, 17],
            [20, 18, 17],
            [40, 45, 35],
            [10, 15, 10],
            [8, 12, 8],
            [15, 15, 12]
        ]
        
        budget = [budget_y1, budget_y2, budget_y3]
        dependances = [(1, 0), (6, 0)]
        exclusions = [(2, 5)]
        
        model = Model("Capital_Budgeting_Energy")
        x = []
        for i in range(n_projets):
            x.append(model.addVar(vtype=GRB.BINARY, name=f"x_{i}"))
        
        model.setObjective(quicksum(van[i] * x[i] for i in range(n_projets)), GRB.MAXIMIZE)
        
        for p in periodes:
            model.addConstr(quicksum(couts[i][p] * x[i] for i in range(n_projets)) <= budget[p], name=f"Budget_P{p+1}")
        
        for (dependant, requis) in dependances:
            model.addConstr(x[dependant] <= x[requis], name=f"Dep_{dependant}_{requis}")
        
        for (proj1, proj2) in exclusions:
            model.addConstr(x[proj1] + x[proj2] <= 1, name=f"Excl_{proj1}_{proj2}")
        
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', 300)
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            projets_selectionnes = []
            cout_total_par_periode = [0] * n_periodes
            
            for i in range(n_projets):
                if x[i].x > 0.5:
                    projets_selectionnes.append({
                        'Projet': projets[i],
                        'VAN (M€)': van[i],
                        'Année 1 (M€)': couts[i][0],
                        'Année 2 (M€)': couts[i][1],
                        'Année 3 (M€)': couts[i][2],
                        'Coût Total (M€)': sum(couts[i])
                    })
                    for p in periodes:
                        cout_total_par_periode[p] += couts[i][p]
            
            df_selected = pd.DataFrame(projets_selectionnes)
            
            cout_total = sum(cout_total_par_periode)
            budget_total = sum(budget)
            van_totale = model.objVal
            
            budget_data = []
            for p in periodes:
                budget_data.append({
                    'Année': f'Année {p+1}',
                    'Utilisé (M€)': cout_total_par_periode[p],
                    'Budget (M€)': budget[p],
                    'Utilisation (%)': f"{(cout_total_par_periode[p]/budget[p])*100:.1f}%"
                })
            df_budget = pd.DataFrame(budget_data)
            
            summary = f"""
## Résultats de l'optimisation

**VAN Totale Maximale:** {van_totale:.2f} M€

**Nombre de projets sélectionnés:** {len(projets_selectionnes)}/{n_projets}

**Coût total:** {cout_total:.0f} M€  
**Budget total disponible:** {budget_total} M€  
**Budget restant:** {budget_total - cout_total:.0f} M€  
**ROI:** {(van_totale/cout_total)*100:.0f}%
"""
            
            # Create visualization
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Analyse de la Sélection d\'Investissements', fontsize=16, fontweight='bold')
            
            # 1. VAN par projet sélectionné
            selected_projects = [projets[i] for i in range(n_projets) if x[i].x > 0.5]
            selected_vans = [van[i] for i in range(n_projets) if x[i].x > 0.5]
            colors1 = plt.cm.viridis([i/len(selected_projects) for i in range(len(selected_projects))])
            bars1 = ax1.barh(selected_projects, selected_vans, color=colors1)
            ax1.set_xlabel('VAN (M€)', fontweight='bold')
            ax1.set_title('VAN par Projet Sélectionné', fontweight='bold')
            ax1.grid(axis='x', alpha=0.3)
            for i, (bar, val) in enumerate(zip(bars1, selected_vans)):
                ax1.text(val, bar.get_y() + bar.get_height()/2, f' {val}M€', 
                        va='center', fontweight='bold')
            
            # 2. Utilisation du budget par année
            years = ['Année 1', 'Année 2', 'Année 3']
            x_pos = range(len(years))
            width = 0.35
            bars_used = ax2.bar([p - width/2 for p in x_pos], cout_total_par_periode, 
                               width, label='Utilisé', color='#e74c3c')
            bars_budget = ax2.bar([p + width/2 for p in x_pos], budget, 
                                 width, label='Budget', color='#3498db', alpha=0.7)
            ax2.set_xlabel('Période', fontweight='bold')
            ax2.set_ylabel('Montant (M€)', fontweight='bold')
            ax2.set_title('Utilisation du Budget par Année', fontweight='bold')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(years)
            ax2.legend()
            ax2.grid(axis='y', alpha=0.3)
            for bars in [bars_used, bars_budget]:
                for bar in bars:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.0f}', ha='center', va='bottom', fontweight='bold')
            
            # 3. Répartition des coûts par projet
            selected_costs = [sum(couts[i]) for i in range(n_projets) if x[i].x > 0.5]
            colors3 = plt.cm.Set3(range(len(selected_projects)))
            wedges, texts, autotexts = ax3.pie(selected_costs, labels=selected_projects, autopct='%1.1f%%',
                                                colors=colors3, startangle=90)
            ax3.set_title('Répartition des Coûts Totaux', fontweight='bold')
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            # 4. Métriques clés
            ax4.axis('off')
            metrics_text = f"""
MÉTRIQUES CLÉS

VAN Totale:        {van_totale:.2f} M€
Coût Total:        {cout_total:.0f} M€
ROI:               {(van_totale/cout_total)*100:.0f}%

Projets:           {len(projets_selectionnes)}/{n_projets} sélectionnés

Budget Total:      {budget_total} M€
Budget Utilisé:    {cout_total:.0f} M€
Budget Restant:    {budget_total - cout_total:.0f} M€
Taux d'utilisation: {(cout_total/budget_total)*100:.1f}%
"""
            ax4.text(0.1, 0.5, metrics_text, fontsize=12, family='monospace',
                    verticalalignment='center', bbox=dict(boxstyle='round', 
                    facecolor='lightblue', alpha=0.3))
            
            plt.tight_layout()
            
            return summary, df_selected, df_budget, "Statut: Solution optimale trouvée", fig
        
        elif model.status == GRB.INFEASIBLE:
            return "Problème infaisable avec ces contraintes", None, None, "Statut: Infaisable", None
        else:
            return f"Statut: {model.status}", None, None, f"Statut: {model.status}", None
            
    except Exception as e:
        return f"Erreur: {str(e)}", None, None, f"Erreur: {str(e)}", None
    finally:
        sys.stdout = sys.__stdout__

def create_problem_9_4_tab():
    with gr.Column():
        gr.Markdown("""
        ## Problème 9.4: Sélection d'Investissements
        ### Secteur Énergétique - Modernisation de Centrales
        
        **Énoncé:** Une compagnie d'énergie doit choisir quelles centrales électriques moderniser pour 
        maximiser la Valeur Actuelle Nette (VAN) totale sans dépasser le budget disponible sur plusieurs périodes.
        
        **Objectif:** MAX Σ(VAN × x)  
        **Type:** PLNE (Binaire)
        
        ### Projets disponibles
        """)
        
        projects_data = {
            'Projet': [
                'Centrale Thermique Nord',
                'Centrale Solaire Sud',
                'Centrale Éolienne Offshore',
                'Centrale Hydroélectrique Est',
                'Centrale Nucléaire Centre',
                'Parc Éolien Terrestre Ouest',
                'Centrale Biomasse Rurale',
                'Centrale Géothermique'
            ],
            'VAN (M€)': [45, 38, 62, 55, 120, 35, 28, 42],
            'Année 1 (M€)': [15, 12, 25, 20, 40, 10, 8, 15],
            'Année 2 (M€)': [20, 15, 20, 18, 45, 15, 12, 15],
            'Année 3 (M€)': [10, 11, 17, 17, 35, 10, 8, 12],
            'Coût Total (M€)': [45, 38, 62, 55, 120, 35, 28, 42]
        }
        df_projects = pd.DataFrame(projects_data)
        gr.Dataframe(value=df_projects, interactive=False)
        
        gr.Markdown("""
        ### Contraintes
        
        **Budget par période:**  
        Σ(coût[i,p] × x[i]) ≤ Budget[p]  ∀p ∈ {1,2,3}
        
        **Dépendances:**
        - Solaire Sud → Thermique Nord
        - Biomasse Rurale → Thermique Nord
        
        **Exclusions mutuelles:**  
        Éolienne Offshore ⊕ Éolien Terrestre
        
        **Variables de décision:**  
        x[i] ∈ {0,1}  ∀i (x[i] = 1 si projet i sélectionné, 0 sinon)
        
        ### Paramètres d'entrée
        """)
        
        with gr.Row():
            budget_y1 = gr.Slider(minimum=50, maximum=150, value=80, step=5, label="Budget Année 1 (M€)")
            budget_y2 = gr.Slider(minimum=50, maximum=150, value=90, step=5, label="Budget Année 2 (M€)")
            budget_y3 = gr.Slider(minimum=50, maximum=150, value=70, step=5, label="Budget Année 3 (M€)")
        
        solve_btn = gr.Button("Résoudre", variant="primary", size="lg")
        
        gr.Markdown("### Résultats")
        
        status_output = gr.Textbox(label="Statut", lines=1)
        summary_output = gr.Markdown()
        
        visualization_output = gr.Plot(label="Analyse Visuelle")
        
        with gr.Row():
            selected_projects = gr.Dataframe(label="Projets sélectionnés", interactive=False)
            budget_usage = gr.Dataframe(label="Utilisation du budget", interactive=False)
        
        with gr.Accordion("Formulation mathématique complète", open=False):
            gr.Markdown("""
```
Maximiser:
Z = 45x₀ + 38x₁ + 62x₂ + 55x₃ + 120x₄ + 35x₅ + 28x₆ + 42x₇

Sous contraintes:
Budget Année 1: 15x₀ + 12x₁ + 25x₂ + 20x₃ + 40x₄ + 10x₅ + 8x₆ + 15x₇ ≤ Budget₁
Budget Année 2: 20x₀ + 15x₁ + 20x₂ + 18x₃ + 45x₄ + 15x₅ + 12x₆ + 15x₇ ≤ Budget₂
Budget Année 3: 10x₀ + 11x₁ + 17x₂ + 17x₃ + 35x₄ + 10x₅ + 8x₆ + 12x₇ ≤ Budget₃

Dépendances:
x₁ ≤ x₀  (Solaire dépend de Thermique)
x₆ ≤ x₀  (Biomasse dépend de Thermique)

Exclusion:
x₂ + x₅ ≤ 1  (Éolienne Offshore et Terrestre s'excluent)

xᵢ ∈ {0,1}  ∀i ∈ {0,1,2,3,4,5,6,7}
```
            """)
        
        solve_btn.click(
            fn=solve_problem_9_4,
            inputs=[budget_y1, budget_y2, budget_y3],
            outputs=[summary_output, selected_projects, budget_usage, status_output, visualization_output]
        )


def create_location_allocation_tab():
    """Crée l'interface Gradio"""
    
    solver = LocationAllocationSolver()
    
    def optimiser_threaded(budget, max_sites, capacite_mult, poids_co2,
                            distance_max):
        """Fonction d'optimisation"""
        
        try:
            resultats, model = solver.solve(
                budget_max=budget,
                max_sites=max_sites,
                capacite_mult=capacite_mult,
                poids_co2=poids_co2,
                distance_max_penalite=distance_max
            )
            
            if resultats['optimal']:
                # Créer visualisation
                fig = solver.visualiser_solution(resultats, 'solution.png')
                plt.close(fig)
                
                # Créer DataFrame des sites
                sites_data = []
                for j in resultats['sites_ouverts']:
                    charge = resultats['charges_sites'][j]['total']
                    capacite = sum(solver.sites[f'capacite_{t}'][j] 
                                  for t in ['recyclable', 'organique', 'dangereux'])
                    taux = (charge / (capacite * capacite_mult)) * 100
                    
                    sites_data.append({
                        'Site': solver.sites['noms'][j],
                        'Coût fixe': f"{solver.sites['cout_fixe'][j]} kTND/an",
                        'Charge': f"{charge:.1f} t/sem",
                        'Capacité': f"{capacite * capacite_mult:.0f} t/sem",
                        'Utilisation': f"{taux:.1f}%",
                        'CO₂': f"{solver.sites['emissions_co2'][j]} t/an"
                    })
                
                df_sites = pd.DataFrame(sites_data)
                
                # Créer DataFrame des affectations
                df_affectations = pd.DataFrame(resultats['affectations'])
                if not df_affectations.empty:
                    df_affectations['proportion'] = df_affectations['proportion'].apply(lambda x: f"{x:.1f}%")
                    df_affectations['demande'] = df_affectations['demande'].apply(lambda x: f"{x:.1f} t")
                
                # Message de résumé
                stats = resultats['statistiques']
                message = f"""
### ✅ OPTIMISATION RÉUSSIE

**💰 Coût Total:** {resultats['cout_total_reel']:.2f} kTND/an
- Coûts fixes: {resultats['cout_fixe']:.2f} kTND/an ({(resultats['cout_fixe']/resultats['cout_total_reel']*100):.1f}%)
- Coûts transport: {resultats['cout_transport']:.2f} kTND/an ({(resultats['cout_transport']/resultats['cout_total_reel']*100):.1f}%)
- Pénalités: {resultats['cout_penalite']:.2f} kTND/an ({(resultats['cout_penalite']/resultats['cout_total_reel']*100):.1f}%)

**🌍 Impact Environnemental:** {resultats['emissions_co2']:.0f} tonnes CO₂/an

**📊 Statistiques:**
- Sites ouverts: {stats['nb_sites_ouverts']}/{solver.n_sites}
- Budget utilisé: {stats['budget_utilise']:.0f}/{budget} kTND/an ({(stats['budget_utilise']/budget*100):.1f}%)
- Taux d'utilisation: {stats['taux_utilisation']:.1f}%
- Demande traitée: {stats['demande_totale']:.0f} tonnes/semaine
- Coût transport moyen: {stats['cout_transport_moyen']:.2f} TND/tonne
"""
                
                return message, df_sites, df_affectations, 'solution.png'
            
            elif resultats['infaisable']:
                return "❌ Problème INFAISABLE - Ajustez les contraintes (augmentez le budget ou le nombre de sites)", None, None, None
            
            else:
                return f"⚠️ Statut: {resultats['status']}", None, None, None
        
        except Exception as e:
            return f"❌ Erreur: {str(e)}", None, None, None
    
    
    # Interface Gradio
    with gr.Column():
        with gr.Blocks(title="Localisation-Allocation Avancé") as interface:
            gr.Markdown("""
            # 🏭 Système de Localisation-Allocation des Centres de Tri
            ## Modèle PLNE/PLM Avancé avec Optimisation Multi-Objectif
            
            **Fonctionnalités:**
            -  Optimisation bi-objectif (coût + CO₂)
            -  Capacités multiples (poids ET volume)
            -  Pénalités de distance
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ Paramètres d'Optimisation")
                    
                    budget = gr.Slider(400, 1300, value=800, step=50,
                                    label="💰 Budget Maximum (kTND/an)")
                    max_sites = gr.Slider(1, 5, value=3, step=1,
                                        label="📍 Nombre Maximum de Sites")
                    capacite_mult = gr.Slider(0.5, 2.0, value=1.0, step=0.1,
                                            label="📦 Multiplicateur de Capacité")
                    poids_co2 = gr.Slider(0, 1, value=0.3, step=0.1,
                                        label="🌍 Poids CO₂ (0=coût, 1=environnement)")
                    distance_max = gr.Slider(20, 50, value=40, step=5,
                                            label="Coût de transport unitaire maximum avant pénalité (TND/t)")
                                    
                    btn_optimiser = gr.Button("🚀 OPTIMISER", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    gr.Markdown("### 📊 Résultats")
                    
                    message_output = gr.Markdown()
                    
                    with gr.Tabs():
                        with gr.Tab("🗺️ Visualisation"):
                            image_output = gr.Image(label="Carte de la Solution")
                        
                        with gr.Tab("📍 Sites"):
                            sites_output = gr.Dataframe(label="Sites Sélectionnés")
                        
                        with gr.Tab("🔄 Affectations"):
                            affectations_output = gr.Dataframe(label="Affectations Détaillées")
            
            btn_optimiser.click(
                fn=optimiser_threaded,
                inputs=[budget, max_sites, capacite_mult, poids_co2,
                        distance_max],
                outputs=[message_output, sites_output, affectations_output, image_output]
            )
            
            gr.Markdown("""
            ---
            ### 📐 Modélisation Mathématique
            
            **Variables:**
            - `y[j] ∈ {0,1}`: 1 si site j ouvert
            - `x[i,j] ∈ [0,1]`: proportion de demande du quartier i, servie par site j
            
            **Fonction Objectif:**
            ```
            MIN Z = (1-λ) × [Σ CoûtFixe[j]×y[j] + Σ CoûtTransport[i,j]×Demande[i]×x[i,j] + Pénalités]
                    + λ × Σ Emissions[j]×y[j]
            ```
            
            **Contraintes principales:**
            1. Desserte complète par type: `Σⱼ x[i,j] = 1` ∀i
            2. Liaison ouverture: `x[i,j] ≤ y[j]` ∀i,j
            3. Capacité poids: `Σᵢ Demande[i]×x[i,j] ≤ Capacité[j]` ∀j
            4. Capacité volume: `Σᵢ Volume[i]×x[i,j] ≤ CapacitéVolume[j]` ∀j
            7. Budget: `Σⱼ CoûtFixe[j]×y[j] ≤ Budget`
            8. Limite sites: `Σⱼ y[j] ≤ MaxSites`
            """)
    
    return interface



def solve_problem_17_2(price_min, price_max, capacity_mult, demand_sens):
    
    try:
        res = problem_17_2.solve(
            price_min=price_min,
            price_max=price_max,
            capacity_multiplier=capacity_mult,
            demand_sensitivity=demand_sens,
            segments=['res', 'comm', 'ind'],
            verbose=False,
            return_dict=True
        )

        if not isinstance(res, dict):
            return "❌ Erreur interne: résultat inattendu du solveur", None, None, "❌ Erreur"

        status = res.get('status')
        if status == 'optimal':
           
            results = res.get('results', [])
            df_results = pd.DataFrame(results)

            
            stats = [
                {'Métrique': '💰 Profit Total', 'Valeur': f"{res.get('objective',0):,.2f} €"},
                {'Métrique': '💵 Revenu Total', 'Valeur': f"{res.get('revenu_total',0):,.2f} €"},
                {'Métrique': '🏭 Coût Total', 'Valeur': f"{res.get('cout_total',0):,.2f} €"},
                {'Métrique': '⚡ Quantité Totale', 'Valeur': f"{res.get('quantite_totale',0):,.0f} kWh"},
                {'Métrique': '📊 Prix Moyen', 'Valeur': f"{res.get('prix_moyen',0):.3f} €/kWh"},
                {'Métrique': '📈 Marge Bénéficiaire', 'Valeur': f"{res.get('marge',0):.1f}%"},
            ]
            df_stats = pd.DataFrame(stats)

         
            summary_md = f"""
        ## ✅ Résultats de l'Optimisation

        - **Profit Total:** {res.get('objective',0):,.2f} €
        - **Revenu Total:** {res.get('revenu_total',0):,.2f} €
        - **Coût Total:** {res.get('cout_total',0):,.2f} €
        - **Quantité Totale:** {res.get('quantite_totale',0):,.0f} kWh
        """
            return summary_md, df_results, df_stats, "✅ Solution optimale trouvée"

        elif status == 'infeasible' or status == 'infeasible_params':
            err = res.get('error', 'Problème infaisable')
            iis = res.get('iis')
            md = f"## ❌ Infaisable\n\n{err}"
            if iis:
                md += f"\n\nIIS écrit: {iis}"
            return md, None, None, "❌ Infaisable"

        else:
            return f"⚠️ Statut: {status}", None, None, f"⚠️ Status: {status}"

    except Exception as e:
        return f"❌ Erreur interne: {e}", None, None, "❌ Erreur"


def create_problem_17_2_tab():
    
    
    with gr.Column():

        gr.Markdown("""
        ## ⚡ Problème 2: Tarification Optimale de l'Électricité
        
        ### 📋 Contexte
        Une compagnie d'électricité doit déterminer les **prix optimaux** pour différentes 
        périodes de la journée afin de **maximiser le profit** tout en respectant les 
        contraintes de capacité de production et en tenant compte de la **demande élastique** 
        des consommateurs.
        
        ### 🎯 Objectif Mathématique
        **Maximiser:** Z = Σ(Prix × Quantité - Coût × Quantité)  
        **Type:** Programmation Linéaire (PL) / Programmation Linéaire Mixte (PLM)
        
        ### 💡 Concept de Demande Élastique
        La demande réagit au prix selon le modèle:  
        **Quantité = Demande de base - Élasticité × Prix**
        
        - Prix ↑ → Demande ↓ (les consommateurs réduisent leur consommation)
        - Prix ↓ → Demande ↑ (les consommateurs consomment plus)
        
        ### 📊 Périodes de la Journée
        """)
      
        periodes_info = {
            'Période': [
                '🌙 Nuit (0h-4h)',
                '🌅 Matin tôt (4h-8h)',
                '☀️ Matin (8h-12h)',
                '🌤️ Après-midi (12h-16h)',
                '🌆 Soirée (16h-20h)',
                '🌃 Nuit tardive (20h-24h)'
            ],
            'Demande de Base': ['5,000 kWh', '8,000 kWh', '12,000 kWh', '10,000 kWh', '15,000 kWh ⚡', '7,000 kWh'],
            'Capacité': ['6,000 kWh', '9,000 kWh', '13,000 kWh', '11,000 kWh', '16,000 kWh', '8,000 kWh'],
            'Coût Production': ['0.05 €/kWh', '0.08 €/kWh', '0.12 €/kWh', '0.10 €/kWh', '0.15 €/kWh', '0.07 €/kWh'],
            'Caractéristique': ['Faible demande', 'Demande croissante', 'Haute demande', 'Demande modérée', 'HEURE DE POINTE', 'Demande décroissante']
        }
        df_periodes = pd.DataFrame(periodes_info)
        gr.Dataframe(value=df_periodes, interactive=False)
        
        gr.Markdown("""
        ### ⚖️ Contraintes du Modèle
        
        #### 1. **Demande Élastique(Relation Prix-Quantité)**
        ```
        q[t] = demande_base[t] - élasticité[t] × p[t]  ∀t
        ```
        La quantité demandée dépend du prix fixé.
        
        #### 2. **Capacité de Production**
        ```
        q[t] ≤ capacité[t]  ∀t
        ```
        Ne peut pas vendre plus que la capacité de production.
        
        #### 3. **Bornes de Prix**
        ```
        prix_min ≤ p[t] ≤ prix_max  ∀t
        ```
        Réglementation des prix (éviter l'abus ou le dumping).
        
        #### 4. **Continuité des Prix**
        ```
        |p[t] - p[t-1]| ≤ 0.15 €/kWh
        ```
        Éviter les chocs de prix entre périodes consécutives.
        
        #### 5. **Non-négativité**
        ```
        p[t] ≥ 0, q[t] ≥ 0  ∀t
        ```
        
        ---
        
        ### 🎛️ Paramètres de Simulation
        
        Ajustez les paramètres pour explorer différents scénarios:
        """)
        
        
        with gr.Row():
            price_min = gr.Slider(
                minimum=0.05, maximum=0.30, value=0.10, step=0.01,
                label="💵 Prix Minimum (€/kWh)",
                info="Prix plancher réglementaire - ne peut pas vendre en dessous"
            )
            price_max = gr.Slider(
                minimum=0.20, maximum=0.80, value=0.50, step=0.05,
                label="💰 Prix Maximum (€/kWh)",
                info="Prix plafond réglementaire - ne peut pas vendre au-dessus"
            )
        
        with gr.Row():
            capacity_mult = gr.Slider(
                minimum=0.5, maximum=1.5, value=1.0, step=0.1,
                label="🏭 Capacité de Production (Multiplicateur)",
                info="1.0 = 100% de capacité | <1.0 = Production réduite | >1.0 = Capacité augmentée"
            )
            demand_sens = gr.Slider(
                minimum=0.5, maximum=2.0, value=1.0, step=0.1,
                label="📊 Sensibilité de la Demande",
                info="1.0 = Normale | >1.0 = Très élastique (réagit plus au prix) | <1.0 = Peu élastique"
            )
        
        gr.Markdown("""
        ### 📝 Scénarios Suggérés
        
        - **Scenario 1 - Standard:** Prix [0.10 - 0.50], Capacité 1.0, Sensibilité 1.0
        - **Scenario 2 - Crise Énergétique:** Prix [0.15 - 0.80], Capacité 0.7, Sensibilité 1.5
        - **Scenario 3 - Surcapacité:** Prix [0.05 - 0.40], Capacité 1.5, Sensibilité 0.8
        - **Scenario 4 - Régulation Stricte:** Prix [0.20 - 0.35], Capacité 1.0, Sensibilité 1.2
        """)

        
        solve_btn = gr.Button("🚀 Optimiser la Tarification", variant="primary", size="lg")
        
        
        gr.Markdown("### 📈 Résultats de l'Optimisation")
        
        status_output = gr.Textbox(label="Statut", lines=1, show_label=True)
        summary_output = gr.Markdown()
        
        with gr.Row():
            with gr.Column(scale=2):
                results_table = gr.Dataframe(
                    label="⚡ Tarification par Période",
                    interactive=False,
                    wrap=True
                )
            with gr.Column(scale=1):
                stats_table = gr.Dataframe(
                    label="📊 Statistiques Globales",
                    interactive=False
                )

        
        with gr.Accordion("📐 Formulation Mathématique Complète", open=False):
            gr.Markdown("""
 ### Modèle d'Optimisation Complet

#### Ensembles et Indices
```
T = {0, 1, 2, 3, 4, 5}  (6 périodes de 4 heures)
t ∈ T : indice de période
```

#### Paramètres
```
a[t]     : Demande de base à la période t (kWh)
b[t]     : Coefficient d'élasticité à la période t
c[t]     : Coût de production à la période t (€/kWh)
Q_max[t] : Capacité de production à la période t (kWh)
p_min    : Prix minimum réglementaire (€/kWh)
p_max    : Prix maximum réglementaire (€/kWh)
Δp_max   : Variation maximale de prix entre périodes (€/kWh)
```

#### Variables de Décision
```
p[t] ∈ R+ : Prix de l'électricité à la période t (€/kWh)
q[t] ∈ R+ : Quantité vendue à la période t (kWh)
```

#### Fonction Objectif
```
Maximiser:
Z = Σ(t∈T) [p[t] × q[t] - c[t] × q[t]]
  = Σ(t∈T) [(p[t] - c[t]) × q[t]]

Où:
- p[t] × q[t] = Revenu à la période t
- c[t] × q[t] = Coût de production à la période t
- (p[t] - c[t]) × q[t] = Profit à la période t
```

#### Contraintes

**1. Demande Élastique (Relation Prix-Quantité):**
```
q[t] = a[t] - b[t] × p[t]    ∀t ∈ T
```

**2. Capacité de Production:**
```
q[t] ≤ Q_max[t]              ∀t ∈ T
```

**3. Bornes de Prix:**
```
p_min ≤ p[t] ≤ p_max         ∀t ∈ T
```

**4. Continuité des Prix (variation entre périodes consécutives):**
```
p[t+1] - p[t] ≤ Δp_max       ∀t ∈ T \ {5}
p[t] - p[t+1] ≤ Δp_max       ∀t ∈ T \ {5}
```

**5. Cycle 24h (continuité entre dernière et première période):**
```
p[0] - p[5] ≤ Δp_max
p[5] - p[0] ≤ Δp_max
```

**6. Non-négativité:**
```
p[t] ≥ 0                     ∀t ∈ T
q[t] ≥ 0                     ∀t ∈ T
```

#### Valeurs Numériques

**Demande de base (kWh):**
```
a = [5000, 8000, 12000, 10000, 15000, 7000]
```

**Élasticité de base:**
```
b = [8000, 12000, 20000, 15000, 25000, 10000] × sensibilité
```

**Capacité (kWh):**
```
Q_max = [6000, 9000, 13000, 11000, 16000, 8000] × multiplicateur
```

**Coûts de production (€/kWh):**
```
c = [0.05, 0.08, 0.12, 0.10, 0.15, 0.07]
```

**Autres paramètres:**
```
Δp_max = 0.15 €/kWh
p_min = 0.10 €/kWh (ajustable)
p_max = 0.50 €/kWh (ajustable)
```

#### Nature du Problème
- **Type:** Programmation Linéaire (PL)
- **Variables:** 12 continues (6 prix + 6 quantités)
- **Contraintes:** ~20 (6 demande + 6 capacité + 2 bornes + ~12 continuité)
- **Complexité:** Polynomial (résolu efficacement par simplex)
            """)
        
      
        
        solve_btn.click(
            fn=solve_problem_17_2,
            inputs=[price_min, price_max, capacity_mult, demand_sens],
            outputs=[summary_output, results_table, stats_table, status_output]
        )

def create_home_tab():
    gr.Markdown("""
    # Optimisation Solver
    ## TP Recherche Opérationnelle - GL3
    
    ### Problèmes implémentés
    
    **Problème 9.4 - Sélection d'Investissements (Capital Budgeting)**
    - Secteur: Énergie
    - Type: PLNE (Binaire)
    - Objectif: Maximiser la VAN totale
    - Contraintes: Budget multi-périodes, dépendances, exclusions
    

    **Problème 17.2 - Tarification Optimale de l'Électricité**
    - Type: PL/PLM
    - Objectif: Déterminer le prix optimal de l'Électricité afin de maximiser le revenu 
    - Contraintes: Demande élastique, Capacité de production, Prix minimum, Prix maximum, Contrainte de continuité       

    **Problèmes 3, 5**
    - À implémenter par les membres de l'équipe

    **Problème 11.4 - Routage du Personnel (VRP)**
    - Type: Vehicle Routing Problem
    - Objectif: Minimiser la distance totale
    - Contraintes: Capacité véhicules, fenêtres temporelles
    
    **Problème - Localisation-Allocation**
    - Type: Facility Location Problem
    - Objectif: Minimiser coûts de transport et d'ouverture
    - Contraintes: Capacité des centres, demande des quartiers

    
    ---
    """)

with gr.Blocks(title="Optimisation - TP RO GL3") as app:
    gr.Markdown("# Optimisation Solver - TP RO GL3")
    
    with gr.Tabs():
        with gr.Tab("Accueil"):
            create_home_tab()
        
        with gr.Tab("Problème 9.4 - Énergie"):
            create_problem_9_4_tab()
        
        with gr.Tab("Problème 4.5 - Localisation"):
            create_location_allocation_tab()
        
        with gr.Tab("Problème 17.2 - Tarification Optimale de l'Électricité"):
            create_problem_17_2_tab()
        
        with gr.Tab("Problème 11.4"):
            create_problem_11_4_tab()
        
        with gr.Tab("Problème 5"):
            gr.Markdown("## Problème 5\nÀ implémenter par membre 5")

if __name__ == "__main__":
    app.launch(share=False, server_name="127.0.0.1", server_port=7860)