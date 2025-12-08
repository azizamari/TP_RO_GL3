from gurobipy import *
import numpy as np

def solve():
    """
    Problème de Localisation-Allocation: Centres de Tri
    Décider où implanter des centres de tri et quels quartiers ils desserviront
    """
    
    # === DONNÉES DU PROBLÈME ===
    
    # Quartiers à desservir (clients)
    quartiers = [
        "Centre-Ville",
        "Zone Industrielle Nord",
        "Quartier Résidentiel Est",
        "Banlieue Sud",
        "Zone Commerciale Ouest",
        "Quartier Universitaire",
        "Zone Portuaire",
        "Quartier Historique"
    ]
    
    # Sites potentiels pour centres de tri
    sites = [
        "Site A - Zone Nord",
        "Site B - Zone Est", 
        "Site C - Zone Sud",
        "Site D - Zone Ouest",
        "Site E - Centre"
    ]
    
    n_quartiers = len(quartiers)
    n_sites = len(sites)
    
    # Demande hebdomadaire de chaque quartier (en tonnes)
    demande = [150, 280, 200, 180, 220, 120, 300, 90]
    
    # Capacité de traitement de chaque site (en tonnes/semaine)
    capacite = [600, 500, 700, 550, 450]
    
    # Coût fixe d'ouverture de chaque site (en k€/an)
    cout_fixe = [250, 200, 280, 220, 300]
    
    # Coût de transport par tonne entre quartier i et site j (en €/tonne)
    # Matrice [quartier][site]
    cout_transport = [
        [15, 25, 35, 30, 10],  # Centre-Ville
        [12, 30, 40, 35, 25],  # Zone Industrielle Nord
        [30, 10, 25, 40, 20],  # Quartier Résidentiel Est
        [35, 30, 12, 25, 30],  # Banlieue Sud
        [30, 40, 28, 10, 25],  # Zone Commerciale Ouest
        [20, 18, 28, 32, 15],  # Quartier Universitaire
        [25, 35, 30, 22, 28],  # Zone Portuaire
        [18, 28, 38, 35, 12]   # Quartier Historique
    ]
    
    # Contraintes de budget et nombre max de sites
    budget_max = 800  # k€/an
    max_sites = 3     # Nombre maximum de sites à ouvrir
    
    # === MODÈLE D'OPTIMISATION ===
    
    model = Model("Localisation_Centres_Tri")
    
    # Variables de décision
    # y[j] = 1 si site j est ouvert, 0 sinon
    y = {}
    for j in range(n_sites):
        y[j] = model.addVar(vtype=GRB.BINARY, name=f"Ouvrir_{j}")
    
    # x[i,j] = proportion de la demande du quartier i servie par le site j
    x = {}
    for i in range(n_quartiers):
        for j in range(n_sites):
            x[i,j] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, 
                                  name=f"Alloc_{i}_{j}")
    
    # Fonction objectif: Minimiser coûts totaux
    # Coûts fixes + Coûts de transport
    cout_total = (
        quicksum(cout_fixe[j] * y[j] for j in range(n_sites)) +
        quicksum(cout_transport[i][j] * demande[i] * x[i,j] 
                for i in range(n_quartiers) 
                for j in range(n_sites)) / 1000  # Conversion en k€
    )
    model.setObjective(cout_total, GRB.MINIMIZE)
    
    # Contraintes
    
    # 1. Chaque quartier doit être entièrement desservi
    for i in range(n_quartiers):
        model.addConstr(
            quicksum(x[i,j] for j in range(n_sites)) == 1,
            name=f"Desserte_Q{i}"
        )
    
    # 2. Un quartier ne peut être servi que par un site ouvert
    for i in range(n_quartiers):
        for j in range(n_sites):
            model.addConstr(
                x[i,j] <= y[j],
                name=f"Ouverture_Q{i}_S{j}"
            )
    
    # 3. Capacité des sites ne doit pas être dépassée
    for j in range(n_sites):
        model.addConstr(
            quicksum(demande[i] * x[i,j] for i in range(n_quartiers)) <= capacite[j],
            name=f"Capacite_S{j}"
        )
    
    # 4. Budget d'investissement
    model.addConstr(
        quicksum(cout_fixe[j] * y[j] for j in range(n_sites)) <= budget_max,
        name="Budget"
    )
    
    # 5. Nombre maximum de sites
    model.addConstr(
        quicksum(y[j] for j in range(n_sites)) <= max_sites,
        name="Max_Sites"
    )
    
    # Paramètres du solveur
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', 600)
    model.optimize()
    
    # === AFFICHAGE DES RÉSULTATS ===
    
    print("\n" + "=" * 90)
    print(" " * 25 + "RÉSULTATS D'OPTIMISATION")
    print("=" * 90)
    
    if model.status == GRB.OPTIMAL:
        print(f"\n{'COÛTS TOTAUX':-^90}")
        cout_fixe_total = sum(cout_fixe[j] * y[j].x for j in range(n_sites))
        cout_transport_total = sum(cout_transport[i][j] * demande[i] * x[i,j].x 
                                   for i in range(n_quartiers) 
                                   for j in range(n_sites)) / 1000
        
        print(f"  Coût fixe d'ouverture:     {cout_fixe_total:>8.2f} k€/an")
        print(f"  Coût de transport:         {cout_transport_total:>8.2f} k€/an")
        print(f"  {'─' * 45}")
        print(f"  COÛT TOTAL:                {model.objVal:>8.2f} k€/an")
        
        print(f"\n{'SITES SÉLECTIONNÉS':-^90}")
        sites_ouverts = []
        for j in range(n_sites):
            if y[j].x > 0.5:
                sites_ouverts.append(j)
                charge = sum(demande[i] * x[i,j].x for i in range(n_quartiers))
                taux_utilisation = (charge / capacite[j]) * 100
                
                print(f"  ✓ {sites[j]:<25} | Coût: {cout_fixe[j]:>6} k€/an | "
                      f"Charge: {charge:>6.1f}/{capacite[j]} t/sem ({taux_utilisation:>5.1f}%)")
        
        print(f"\n  Nombre de sites ouverts: {len(sites_ouverts)}/{n_sites}")
        print(f"  Budget utilisé: {cout_fixe_total:.0f}/{budget_max} k€/an")
        
        print(f"\n{'AFFECTATION DES QUARTIERS':-^90}")
        print(f"  {'Quartier':<30} | {'Site assigné':<25} | Demande | Distance")
        print(f"  {'-' * 88}")
        
        for i in range(n_quartiers):
            for j in range(n_sites):
                if x[i,j].x > 0.01:  # Seuil pour éviter les arrondis
                    proportion = x[i,j].x * 100
                    if proportion > 99:  # Affectation complète
                        print(f"  {quartiers[i]:<30} | {sites[j]:<25} | "
                              f"{demande[i]:>4} t  | {cout_transport[i][j]:>3} €/t")
                    else:  # Affectation partielle
                        demande_partielle = demande[i] * x[i,j].x
                        print(f"  {quartiers[i]:<30} | {sites[j]:<25} | "
                              f"{demande_partielle:>4.0f} t ({proportion:.0f}%) | {cout_transport[i][j]:>3} €/t")
        
        print(f"\n{'STATISTIQUES':-^90}")
        demande_totale = sum(demande)
        capacite_totale = sum(capacite[j] for j in sites_ouverts)
        print(f"  Demande totale à traiter:  {demande_totale:>6} tonnes/semaine")
        print(f"  Capacité totale installée: {capacite_totale:>6} tonnes/semaine")
        print(f"  Taux d'utilisation global: {(demande_totale/capacite_totale)*100:>6.1f}%")
        
        # Distance moyenne pondérée
        distance_moy = sum(cout_transport[i][j] * demande[i] * x[i,j].x 
                          for i in range(n_quartiers) 
                          for j in range(n_sites)) / demande_totale
        print(f"  Coût de transport moyen:   {distance_moy:>6.2f} €/tonne")
        
    elif model.status == GRB.INFEASIBLE:
        print("\n⚠️  PROBLÈME INFAISABLE")
        print("  Les contraintes ne peuvent pas être satisfaites simultanément.")
        print("  Suggestions:")
        print("  - Augmenter le budget disponible")
        print("  - Augmenter le nombre maximum de sites autorisés")
        print("  - Vérifier les capacités des sites")
        model.computeIIS()
        print("\n  Contraintes en conflit sauvegardées dans 'location_iis.ilp'")
        model.write("location_iis.ilp")
        
    elif model.status == GRB.TIME_LIMIT:
        print("\n⚠️  LIMITE DE TEMPS ATTEINTE")
        print(f"  Meilleure solution trouvée: {model.objVal:.2f} k€/an")
        
    else:
        print(f"\n⚠️  STATUT: {model.status}")
    
    print("=" * 90)
    
    # Sauvegarde du modèle
    model.write("localisation_centres_tri.lp")
    print("\n📄 Modèle sauvegardé: localisation_centres_tri.lp\n")
    
    return model


if __name__ == "__main__":
    solve()