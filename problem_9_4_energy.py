from gurobipy import *

def solve():
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
    
    budget = [80, 90, 70]
    dependances = [(1, 0), (6, 0)]
    exclusions = [(2, 5)]
    
    model = Model("Capital_Budgeting_Energy")
    x = []
    for i in range(n_projets):
        x.append(model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{projets[i][:10]}"))
    
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
    
    print("\n" + "=" * 80)
    print("RÉSULTATS")
    print("=" * 80)
    
    if model.status == GRB.OPTIMAL:
        print(f"\nVAN totale: {model.objVal:.2f} M€")
        print(f"\nProjets sélectionnés:")
        projets_selectionnes = []
        van_totale = 0
        cout_total_par_periode = [0] * n_periodes
        
        for i in range(n_projets):
            if x[i].x > 0.5:
                projets_selectionnes.append(i)
                van_totale += van[i]
                print(f"  {projets[i]} - VAN: {van[i]} M€")
                for p in periodes:
                    cout_total_par_periode[p] += couts[i][p]
        
        print(f"\nNombre de projets: {len(projets_selectionnes)}/{n_projets}")
        print(f"\nBudget utilisé par année:")
        for p in periodes:
            print(f"  Année {p+1}: {cout_total_par_periode[p]:.0f}/{budget[p]} M€")
        cout_total = sum(cout_total_par_periode)
        budget_total = sum(budget)
        print(f"\nCoût total: {cout_total:.0f} M€")
        print(f"Budget restant: {budget_total - cout_total:.0f} M€")
        print(f"ROI: {(van_totale/cout_total)*100:.0f}%")
        
    elif model.status == GRB.INFEASIBLE:
        print("\nProblème infaisable")
        model.computeIIS()
        model.write("model_iis.ilp")
    elif model.status == GRB.UNBOUNDED:
        print("\nProblème non borné")
    else:
        print(f"\nStatut: {model.status}")
    
    print("=" * 80)
    model.write("capital_budgeting_energy.lp")
    
    return model

if __name__ == "__main__":
    solve()
