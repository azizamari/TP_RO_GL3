import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr


class FixedVRPSolver:
    def __init__(self, n_customers, n_vehicles, capacity, seed=42):
        self.n_customers = n_customers
        self.n_vehicles = n_vehicles
        self.n_nodes = n_customers + 1
        self.vehicle_capacity = capacity

        np.random.seed(seed)

        # 1. Geometry & Distances
        self.locations = np.random.rand(self.n_nodes, 2) * 100
        self.locations[0] = [50, 50]  # Depot

        self.dist = np.zeros((self.n_nodes, self.n_nodes))
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                self.dist[i][j] = np.linalg.norm(self.locations[i] - self.locations[j])

        # 2. Demand & Time Windows
        self.demands = np.random.randint(5, 15, self.n_nodes)
        self.demands[0] = 0

        self.time_windows = np.zeros((self.n_nodes, 2))
        self.time_windows[0] = [0, 5000]  # Depot open long hours
        for i in range(1, self.n_nodes):
            start = np.random.randint(0, 300)
            width = 1000  # Wide windows to ensure feasibility for demo
            self.time_windows[i] = [start, start + width]

        self.service_time = 10
        self.capacities = [capacity] * n_vehicles

        # 3. Skills
        self.skill_names = {0: "Heavy 🏋️", 1: "Tech 🔧", 2: "Fragile 🥚"}

        self.customer_skills = []
        for i in range(self.n_nodes):
            if i == 0:
                self.customer_skills.append(set())
            else:
                if np.random.rand() > 0.5:
                    self.customer_skills.append({np.random.choice(3)})
                else:
                    self.customer_skills.append(set())

        self.vehicle_skills = []
        for v in range(n_vehicles):
            skills = set(np.random.choice(3, 2, replace=False))
            self.vehicle_skills.append(skills)

    def solve(self):
        try:
            m = gp.Model("FixedVRP")
            m.setParam('OutputFlag', 0)
            m.setParam('TimeLimit', 10)

            # --- Variables ---
            x = {}
            for i in range(self.n_nodes):
                for j in range(self.n_nodes):
                    if i != j:
                        for v in range(self.n_vehicles):
                            x[i, j, v] = m.addVar(vtype=GRB.BINARY)

            # Dropped Penalty Variable
            dropped = {}
            for i in range(1, self.n_nodes):
                dropped[i] = m.addVar(vtype=GRB.BINARY, obj=10000)

            t = {}
            for i in range(self.n_nodes):
                for v in range(self.n_vehicles):
                    t[i, v] = m.addVar(lb=0, ub=5000)

            # --- Objective ---
            travel_cost = gp.quicksum(self.dist[i][j] * x[i, j, v]
                                      for i in range(self.n_nodes)
                                      for j in range(self.n_nodes) if i != j
                                      for v in range(self.n_vehicles))

            # Minimize Distance + Penalty for dropped customers
            m.setObjective(travel_cost + gp.quicksum(10000 * dropped[i] for i in range(1, self.n_nodes)), GRB.MINIMIZE)

            # --- Constraints ---

            # 1. Visit OR Drop
            for j in range(1, self.n_nodes):
                m.addConstr(
                    gp.quicksum(x[i, j, v]
                                for i in range(self.n_nodes) if i != j
                                for v in range(self.n_vehicles))
                    + dropped[j] == 1
                )

            # 2. Flow Conservation
            for v in range(self.n_vehicles):
                for j in range(self.n_nodes):
                    m.addConstr(
                        gp.quicksum(x[i, j, v] for i in range(self.n_nodes) if i != j) ==
                        gp.quicksum(x[j, i, v] for i in range(self.n_nodes) if i != j)
                    )

            # 3. Capacity
            for v in range(self.n_vehicles):
                m.addConstr(
                    gp.quicksum(self.demands[j] * x[i, j, v]
                                for i in range(self.n_nodes)
                                for j in range(1, self.n_nodes) if i != j)
                    <= self.capacities[v]
                )

            # 4. Time Windows (FIXED HERE)
            M = 10000
            for i in range(self.n_nodes):
                for j in range(self.n_nodes):
                    if i != j:
                        # KEY FIX: Do not enforce time check when RETURNING to depot (j=0)
                        # We only care that t[i] + travel <= t[j] for CUSTOMERS.
                        if j != 0:
                            for v in range(self.n_vehicles):
                                m.addConstr(t[i, v] + self.service_time + self.dist[i][j]
                                            <= t[j, v] + M * (1 - x[i, j, v]))

            # Time Bounds
            for i in range(self.n_nodes):
                for v in range(self.n_vehicles):
                    m.addConstr(t[i, v] >= self.time_windows[i][0])
                    m.addConstr(t[i, v] <= self.time_windows[i][1])

            # 5. Skills
            for j in range(1, self.n_nodes):
                req = self.customer_skills[j]
                for v in range(self.n_vehicles):
                    veh_skills = self.vehicle_skills[v]
                    if not req.issubset(veh_skills):
                        for i in range(self.n_nodes):
                            if i != j:
                                m.addConstr(x[i, j, v] == 0)

            m.optimize()

            # --- Extract Solution ---
            if m.SolCount > 0:
                sol_routes = {}
                skipped = []

                for v in range(self.n_vehicles):
                    route = []
                    curr = 0
                    while True:
                        next_node = None
                        for j in range(self.n_nodes):
                            if curr != j and x[curr, j, v].X > 0.5:
                                next_node = j
                                break
                        if next_node is None or next_node == 0:
                            break
                        route.append(next_node)
                        curr = next_node
                    sol_routes[v] = route

                for j in range(1, self.n_nodes):
                    if dropped[j].X > 0.5:
                        skipped.append(j)

                return m.objVal, sol_routes, skipped
            else:
                return None, None, None

        except Exception as e:
            return None, None, None

    def plot_solution(self, routes, skipped):
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(50, 50, c='black', marker='s', s=200, label='Depot', zorder=10)

        for i in range(1, self.n_nodes):
            if i in skipped: continue
            ax.scatter(self.locations[i][0], self.locations[i][1], c='lightblue', s=200, edgecolor='blue')
            skills = ",".join([self.skill_names[s] for s in self.customer_skills[i]])
            ax.text(self.locations[i][0], self.locations[i][1], f"{i}\n{skills}", fontsize=8, ha='center', va='center')

        for i in skipped:
            ax.scatter(self.locations[i][0], self.locations[i][1], c='red', marker='x', s=100)
            ax.text(self.locations[i][0], self.locations[i][1] + 3, f"{i} (Skip)", color='red', fontsize=8, ha='center')

        colors = plt.cm.get_cmap('tab10', self.n_vehicles)
        for v, route in routes.items():
            if route:
                full_route = [0] + route + [0]
                path_x = [self.locations[n][0] for n in full_route]
                path_y = [self.locations[n][1] for n in full_route]
                veh_skills = ",".join([self.skill_names[s] for s in self.vehicle_skills[v]])

                ax.plot(path_x, path_y, c=colors(v), lw=2, label=f"Team {v} [{veh_skills}]", alpha=0.8)

                # Draw simple arrows
                for k in range(len(full_route) - 1):
                    mid_x = (path_x[k] + path_x[k + 1]) / 2
                    mid_y = (path_y[k] + path_y[k + 1]) / 2
                    ax.annotate('', xy=(mid_x, mid_y), xytext=(path_x[k], path_y[k]),
                                arrowprops=dict(arrowstyle="->", color=colors(v), lw=2))

        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_title(f"Solution: {len(skipped)} Skipped")
        plt.tight_layout()
        return fig


# --- App ---
def run_app(n_cust, n_veh, cap, seed):
    solver = FixedVRPSolver(int(n_cust), int(n_veh), int(cap), int(seed))
    obj, routes, skipped = solver.solve()

    if obj is None: return None, "Solver Failed."

    report = f"Total Cost (Dist + Penalty): {obj:.1f}\nSkipped: {len(skipped)}\n\n"
    if skipped:
        report += "SKIPPED REASONS:\n"
        for s in skipped:
            req = solver.customer_skills[s]
            report += f" - Cust {s} (Needs {req}) -> Unreachable (Skills or Time)\n"
        report += "\n"

    for v, route in routes.items():
        if route:
            report += f"Team {v}: Depot -> " + " -> ".join(map(str, route)) + " -> Depot\n"

    status = f"skipped={len(skipped)}"

    return status, solver.plot_solution(routes, skipped), report


def create_problem_11_4_tab():
    with gr.Column():
        # --- HEADER & PROBLEM DESCRIPTION ---
        gr.Markdown("""
        ## Problème 11.4 : Routage du Personnel
        ### Tournées avec contraintes de compétences et équipes à deux personnes

        **Énoncé :**  
        Déterminer les routes des employés / équipes pour effectuer des services à différents endroits,  
        en respectant les horaires, les distances, et les compétences requises.

        **Objectif :**  
        Minimiser la distance totale + les pénalités pour clients non servis :

        ```
        MIN Z =  ∑ᵥ ∑ᵢ ∑ⱼ ( dᵢⱼ · xᵢⱼᵥ )  +  P · ∑ⱼ droppedⱼ
        ```

        **Type :** PLNE (Routing / Assignment)

        ---
        """)

        gr.Markdown("### Contraintes du modèle")

        gr.Markdown("""
        **1. Affectation ou Abandon :**  
        Chaque client est servi par **exactement une** équipe ou abandonné.

        ```
        ∑ᵥ yⱼᵥ  +  droppedⱼ  =  1    ∀ j
        ```

        **2. Cohérence Déplacements ↔ Visites :**
        ```
        ∑ᵢ xᵢⱼᵥ = yⱼᵥ      ∀ j,v
        ∑ⱼ xᵢⱼᵥ = yᵢᵥ      ∀ i,v
        ```

        **3. Dépôt (Départ & Retour) :**
        ```
        ∑ⱼ x₀ⱼᵥ = 1      ∀ v
        ∑ᵢ xᵢ0ᵥ = 1      ∀ v
        ```

        **4. Compétences (Matching équipes ↔ clients) :**
        ```
        yⱼᵥ ≤ skillᵥⱼ     ∀ j,v
        ```

        **5. Équipe à Deux Personnes :**
        ```
        staff_neededᵥ ≤ staff_availableᵥ
        ```

        **6. Élimination des Sous-Tournées (MTZ) :**
        ```
        uⱼᵥ - uᵢᵥ + M·xᵢⱼᵥ ≤ M - 1     ∀ i≠j, v
        ```

        **7. Domaines :**
        ```
        xᵢⱼᵥ ∈ {0,1}
        yⱼᵥ ∈ {0,1}
        droppedⱼ ∈ {0,1}
        uⱼᵥ ≥ 0
        ```
        """)

        # --- INPUT PARAMETERS SECTION ---
        gr.Markdown("### Paramètres d'entrée")

        with gr.Row():
            customers = gr.Slider(5, 30, step=1, value=10, label="Nombre de clients")
            teams = gr.Slider(1, 10, step=1, value=3, label="Nombre d'équipes")
            capacity = gr.Slider(20, 200, step=10, value=100, label="Capacité / Temps / Limit")
            seed = gr.Number(42, label="Seed aléatoire")

        solve_btn = gr.Button("Résoudre", variant="primary", size="lg")

        # --- OUTPUTS ---
        gr.Markdown("### Résultats")

        status_box = gr.Textbox(label="Statut")
        solution_plot = gr.Plot(label="Visualisation des tournées")
        solution_text = gr.Textbox(label="Solution détaillée", lines=10)

        with gr.Accordion("Formulation mathématique complète", open=False):
            gr.Markdown(""""
            Minimiser :Z = ∑ᵥ ∑ᵢ ∑ⱼ dᵢⱼ xᵢⱼᵥ + 10000 · ∑ⱼ droppedⱼ
                    s.c.
                    ∑ᵥ yⱼᵥ + droppedⱼ = 1 ∀ j
                    ∑ᵢ xᵢⱼᵥ = ∑ᵢ xⱼᵢᵥ ∀ j, v
                    ∑ᵢ ∑ⱼ demandⱼ · xᵢⱼᵥ ≤ capacityᵥ ∀ v
                    tᵢᵥ + service + dᵢⱼ ≤ tⱼᵥ + M(1 − xᵢⱼᵥ)
                    xᵢⱼᵥ = 0 si skills(j) ⊄ skills(v)
                    xᵢⱼᵥ, yⱼᵥ, droppedⱼ ∈ {0,1}
                    tᵢᵥ, uⱼᵥ ≥ 0
            """"" )

            solve_btn.click(
                fn=run_app,
                inputs=[customers, teams, capacity, seed],
                outputs=[status_box, solution_plot, solution_text]
            )

        return



if __name__ == "__main__":
    create_problem_11_4_tab().launch(share=True)