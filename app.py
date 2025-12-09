import gradio as gr
import pandas as pd
from gurobipy import *
from problem_11_4_routage_du_personnel import *
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

def create_home_tab():
    gr.Markdown("""
    # Optimisation Solver
    ## TP Recherche Opérationnelle - GL3
    
    ### Problèmes disponibles
    
    **Problème 9.4 - Sélection d'Investissements (Énergie)**
    - Type: PLNE (Binaire)
    - Objectif: Maximiser la VAN totale
    - Contraintes: Budget multi-périodes, dépendances, exclusions
    
    **Problème 11.4 - Routage du personnel **
    - Type: PLNE/PLM
    - Objectif: optimiser le distribution: Tournées des équipes de livraison à deux personnes(minimiser la distance totale parcourue et minimiser le nombre de clients non servis).
    - Contraintes: Routage principal et Demande(Visiter ou abandonner, Conservation du Flux, Limite de Capacité), Planification et Fenêtres Temporelles(Bornes des Fenêtres, Séquencement Temporel (Big M)), Appariement des Compétences
    
    **Problèmes 1, 2, 3, 5**
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
        
        with gr.Tab("Problème 1"):
            gr.Markdown("## Problème 1\nÀ implémenter par membre 1")
        
        with gr.Tab("Problème 2"):
            gr.Markdown("## Problème 2\nÀ implémenter par membre 2")
        
        with gr.Tab("Problème 11.4"):
            #gr.Markdown("## Problème 3\nÀ implémenter par membre 3")
            create_problem_11_4_tab()
        
        with gr.Tab("Problème 5"):
            gr.Markdown("## Problème 5\nÀ implémenter par membre 5")

if __name__ == "__main__":
    app.launch(share=False, server_name="127.0.0.1", server_port=7860)
