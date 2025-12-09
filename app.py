import gradio as gr
import pandas as pd
from gurobipy import *
import io
import sys

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
            
            return summary, df_selected, df_budget, "Statut: Solution optimale trouvée"
        
        elif model.status == GRB.INFEASIBLE:
            return "Problème infaisable avec ces contraintes", None, None, "Statut: Infaisable"
        else:
            return f"Statut: {model.status}", None, None, f"Statut: {model.status}"
            
    except Exception as e:
        return f"Erreur: {str(e)}", None, None, f"Erreur: {str(e)}"
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
            outputs=[summary_output, selected_projects, budget_usage, status_output]
        )

def solve_location_allocation(budget_max, max_sites, capacite_multiplicateur):
    """Résout le problème de localisation-allocation des centres de tri"""
    output = io.StringIO()
    sys.stdout = output
    
    try:
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
        
        sites = [
            "Site A - Zone Nord",
            "Site B - Zone Est", 
            "Site C - Zone Sud",
            "Site D - Zone Ouest",
            "Site E - Centre"
        ]
        
        n_quartiers = len(quartiers)
        n_sites = len(sites)
        
        demande = [150, 280, 200, 180, 220, 120, 300, 90]
        capacite_base = [600, 500, 700, 550, 450]
        capacite = [int(c * capacite_multiplicateur) for c in capacite_base]
        cout_fixe = [250, 200, 280, 220, 300]
        
        cout_transport = [
            [15, 25, 35, 30, 10],
            [12, 30, 40, 35, 25],
            [30, 10, 25, 40, 20],
            [35, 30, 12, 25, 30],
            [30, 40, 28, 10, 25],
            [20, 18, 28, 32, 15],
            [25, 35, 30, 22, 28],
            [18, 28, 38, 35, 12]
        ]
        
        model = Model("Localisation_Centres_Tri")
        
        y = {}
        for j in range(n_sites):
            y[j] = model.addVar(vtype=GRB.BINARY, name=f"Site_{j}")
        
        x = {}
        for i in range(n_quartiers):
            for j in range(n_sites):
                x[i,j] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, 
                                      name=f"Alloc_{i}_{j}")
        
        cout_total = (
            quicksum(cout_fixe[j] * y[j] for j in range(n_sites)) +
            quicksum(cout_transport[i][j] * demande[i] * x[i,j] 
                    for i in range(n_quartiers) 
                    for j in range(n_sites)) / 1000
        )
        model.setObjective(cout_total, GRB.MINIMIZE)
        
        for i in range(n_quartiers):
            model.addConstr(
                quicksum(x[i,j] for j in range(n_sites)) == 1,
                name=f"Desserte_{i}"
            )
        
        for i in range(n_quartiers):
            for j in range(n_sites):
                model.addConstr(x[i,j] <= y[j], name=f"Lien_{i}_{j}")
        
        for j in range(n_sites):
            model.addConstr(
                quicksum(demande[i] * x[i,j] for i in range(n_quartiers)) <= capacite[j],
                name=f"Cap_{j}"
            )
        
        model.addConstr(
            quicksum(cout_fixe[j] * y[j] for j in range(n_sites)) <= budget_max,
            name="Budget"
        )
        
        model.addConstr(
            quicksum(y[j] for j in range(n_sites)) <= max_sites,
            name="Max_Sites"
        )
        
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', 300)
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            sites_ouverts = []
            sites_data = []
            
            cout_fixe_total = sum(cout_fixe[j] * y[j].x for j in range(n_sites))
            cout_transport_total = sum(cout_transport[i][j] * demande[i] * x[i,j].x 
                                       for i in range(n_quartiers) 
                                       for j in range(n_sites)) / 1000
            
            for j in range(n_sites):
                if y[j].x > 0.5:
                    sites_ouverts.append(j)
                    charge = sum(demande[i] * x[i,j].x for i in range(n_quartiers))
                    taux = (charge / capacite[j]) * 100
                    
                    sites_data.append({
                        'Site': sites[j],
                        'Coût fixe (k€/an)': cout_fixe[j],
                        'Capacité (t/sem)': capacite[j],
                        'Charge (t/sem)': f"{charge:.1f}",
                        'Utilisation': f"{taux:.1f}%"
                    })
            
            df_sites = pd.DataFrame(sites_data)
            
            affectations = []
            for i in range(n_quartiers):
                for j in range(n_sites):
                    if x[i,j].x > 0.01:
                        proportion = x[i,j].x * 100
                        demande_allouee = demande[i] * x[i,j].x
                        
                        affectations.append({
                            'Quartier': quartiers[i],
                            'Site': sites[j],
                            'Demande (t)': f"{demande_allouee:.1f}",
                            'Proportion': f"{proportion:.1f}%",
                            'Coût (€/t)': cout_transport[i][j]
                        })
            
            df_affectations = pd.DataFrame(affectations)
            
            demande_totale = sum(demande)
            capacite_totale = sum(capacite[j] for j in sites_ouverts)
            distance_moy = sum(cout_transport[i][j] * demande[i] * x[i,j].x 
                              for i in range(n_quartiers) 
                              for j in range(n_sites)) / demande_totale
            
            summary = f"""
## Résultats de l'optimisation

### 💰 Coûts
**Coût total annuel:** {model.objVal:.2f} k€/an
- Coûts fixes: {cout_fixe_total:.2f} k€/an
- Coûts de transport: {cout_transport_total:.2f} k€/an

### 📍 Sites
**Sites ouverts:** {len(sites_ouverts)}/{n_sites}
**Budget utilisé:** {cout_fixe_total:.0f}/{budget_max} k€/an ({(cout_fixe_total/budget_max)*100:.1f}%)

### 📊 Statistiques
**Demande totale:** {demande_totale} t/semaine
**Capacité installée:** {capacite_totale} t/semaine
**Taux d'utilisation:** {(demande_totale/capacite_totale)*100:.1f}%
**Coût de transport moyen:** {distance_moy:.2f} €/tonne
"""
            
            return summary, df_sites, df_affectations, "✓ Solution optimale trouvée"
        
        elif model.status == GRB.INFEASIBLE:
            return "❌ Problème infaisable - Ajustez les contraintes", None, None, "Infaisable"
        else:
            return f"Statut: {model.status}", None, None, f"Statut: {model.status}"
            
    except Exception as e:
        return f"Erreur: {str(e)}", None, None, f"Erreur: {str(e)}"
    finally:
        sys.stdout = sys.__stdout__


def create_location_allocation_tab():
    with gr.Column():
        gr.Markdown("""
        ## Problème de Localisation-Allocation
        ### Centres de Tri et Affectation des Quartiers
        
        **Énoncé:** Une municipalité doit décider où implanter des centres de tri pour traiter 
        les déchets de différents quartiers, en minimisant les coûts totaux (coûts fixes d'ouverture 
        + coûts de transport).
        
        **Objectif:** MIN Σ(Coûts fixes × y) + Σ(Coûts transport × Demande × x)  
        **Type:** PLNE / PLM (Mixte Binaire-Continu)
        
        ### Données du problème
        """)
        
        quartiers_data = {
            'Quartier': [
                'Centre-Ville', 'Zone Industrielle Nord', 'Quartier Résidentiel Est',
                'Banlieue Sud', 'Zone Commerciale Ouest', 'Quartier Universitaire',
                'Zone Portuaire', 'Quartier Historique'
            ],
            'Demande (t/semaine)': [150, 280, 200, 180, 220, 120, 300, 90]
        }
        df_quartiers = pd.DataFrame(quartiers_data)
        
        sites_data = {
            'Site': [
                'Site A - Zone Nord', 'Site B - Zone Est', 'Site C - Zone Sud',
                'Site D - Zone Ouest', 'Site E - Centre'
            ],
            'Capacité (t/sem)': [600, 500, 700, 550, 450],
            'Coût fixe (k€/an)': [250, 200, 280, 220, 300]
        }
        df_sites = pd.DataFrame(sites_data)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("**Quartiers à desservir**")
                gr.Dataframe(value=df_quartiers, interactive=False)
            with gr.Column():
                gr.Markdown("**Sites potentiels**")
                gr.Dataframe(value=df_sites, interactive=False)
        
        gr.Markdown("""
        ### Modélisation mathématique
        
        **Variables de décision:**
        - `y[j] ∈ {0,1}` : 1 si site j est ouvert, 0 sinon
        - `x[i,j] ∈ [0,1]` : proportion de la demande du quartier i servie par site j
        
        **Fonction objectif:**
        ```
        Minimiser Z = Σⱼ (CoûtFixe[j] × y[j]) + Σᵢ Σⱼ (CoûtTransport[i,j] × Demande[i] × x[i,j])
        ```
        
        **Contraintes:**
        1. **Desserte complète:** `Σⱼ x[i,j] = 1` ∀i (chaque quartier entièrement desservi)
        2. **Liaison ouverture:** `x[i,j] ≤ y[j]` ∀i,j (service uniquement par sites ouverts)
        3. **Capacité:** `Σᵢ (Demande[i] × x[i,j]) ≤ Capacité[j]` ∀j
        4. **Budget:** `Σⱼ (CoûtFixe[j] × y[j]) ≤ Budget`
        5. **Limite sites:** `Σⱼ y[j] ≤ MaxSites`
        
        ### Paramètres de simulation
        """)
        
        with gr.Row():
            budget_slider = gr.Slider(
                minimum=400, maximum=1200, value=800, step=50,
                label="Budget maximum (k€/an)"
            )
            max_sites_slider = gr.Slider(
                minimum=2, maximum=5, value=3, step=1,
                label="Nombre maximum de sites"
            )
            capacite_slider = gr.Slider(
                minimum=0.5, maximum=1.5, value=1.0, step=0.1,
                label="Multiplicateur de capacité"
            )
        
        solve_btn = gr.Button("🚀 Résoudre", variant="primary", size="lg")
        
        gr.Markdown("### Résultats")
        
        status_output = gr.Textbox(label="Statut", lines=1)
        summary_output = gr.Markdown()
        
        with gr.Row():
            sites_output = gr.Dataframe(label="Sites sélectionnés", interactive=False)
            affectations_output = gr.Dataframe(label="Affectation des quartiers", interactive=False)
        
        with gr.Accordion("💡 Interprétation des résultats", open=False):
            gr.Markdown("""
            **Comment lire les résultats:**
            
            - **Coût total:** Somme des coûts fixes (ouverture des sites) et coûts variables (transport)
            - **Taux d'utilisation:** Indique si les capacités sont bien utilisées (optimal entre 70-90%)
            - **Affectations partielles:** Un quartier peut être desservi par plusieurs sites si économique
            - **Coût de transport moyen:** Plus il est bas, meilleure est la localisation
            
            **Optimisations possibles:**
            - Augmenter le budget si infaisable
            - Augmenter les capacités si sites surchargés
            - Réduire le nombre de sites pour économiser sur les coûts fixes
            """)
        
        solve_btn.click(
            fn=solve_location_allocation,
            inputs=[budget_slider, max_sites_slider, capacite_slider],
            outputs=[summary_output, sites_output, affectations_output, status_output]
        )

def create_home_tab():
    gr.Markdown("""
    # Optimisation Solver
    ## TP Recherche Opérationnelle - GL3
    
    ### Problèmes disponibles
    
    **Problème 9.4 - Sélection d'Investissements (Énergie)**
    - Type: PLNE (Binaire)
    - Objectif: Maximiser la VAN totale
    - Contraintes: Budget multi-périodes, dépendances, exclusions
                
    **Problème 4.5 - Localisation-Allocation (Centres de Tri)**
    - Type: PLNE/PLM (Mixte Binaire-Continu)
    - Objectif: Minimiser coûts totaux (fixes + transport)
    - Contraintes: Budget, capacités, desserte complète
    
    **Problèmes 2, 3, 5**
    - À implémenter par les membres de l'équipe
    
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
        
        with gr.Tab("Problème 2"):
            gr.Markdown("## Problème 2\nÀ implémenter par membre 2")
        
        with gr.Tab("Problème 3"):
            gr.Markdown("## Problème 3\nÀ implémenter par membre 3")
        
        with gr.Tab("Problème 5"):
            gr.Markdown("## Problème 5\nÀ implémenter par membre 5")

if __name__ == "__main__":
    app.launch(share=False, server_name="127.0.0.1", server_port=7860)
