import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt

"""
=============================================================================
STEP 1: CLASS DEFINITION & DATA GENERATION
=============================================================================
This class creates a VRP problem instance with random data
"""


class VRPTimWindowsSkills:
    def __init__(self, n_customers, n_vehicles):
        """
        Initialize the VRP problem

        Parameters:
        - n_customers: Number of delivery/service locations
        - n_vehicles: Number of delivery teams
        """
        self.n_customers = n_customers
        self.n_vehicles = n_vehicles
        self.n_nodes = n_customers + 1  # +1 for depot (node 0)

        # Set random seed for reproducibility
        np.random.seed(42)

        # =====================================================================
        # STEP 1.1: Generate Node Locations (Geographic Coordinates)
        # =====================================================================
        # Create random (x,y) coordinates for all locations
        self.locations = np.random.rand(self.n_nodes, 2) * 100
        self.locations[0] = [50, 50]  # Depot at center

        print(f"✓ Generated {self.n_nodes} locations (1 depot + {n_customers} customers)")

        # =====================================================================
        # STEP 1.2: Calculate Distance Matrix
        # =====================================================================
        # Distance from every location to every other location
        # dist[i][j] = Euclidean distance from location i to location j
        self.dist = np.zeros((self.n_nodes, self.n_nodes))
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                self.dist[i][j] = np.linalg.norm(
                    self.locations[i] - self.locations[j]
                )

        print(f"✓ Calculated {self.n_nodes}x{self.n_nodes} distance matrix")

        # =====================================================================
        # STEP 1.3: Customer Demands
        # =====================================================================
        # How many units (packages, kg, etc.) each customer needs
        self.demands = np.random.randint(5, 20, self.n_nodes)
        self.demands[0] = 0  # Depot has no demand

        print(f"✓ Generated customer demands: {self.demands[1:6]}... (showing first 5)")

        # =====================================================================
        # STEP 1.4: Vehicle Capacities
        # =====================================================================
        # Maximum load each vehicle can carry
        self.capacities = [80] * n_vehicles

        print(f"✓ Vehicle capacity: {self.capacities[0]} units each")

        # =====================================================================
        # STEP 1.5: Time Windows
        # =====================================================================
        # [earliest_time, latest_time] when service can start
        # Example: Customer available between 10:00 and 14:00
        self.time_windows = np.zeros((self.n_nodes, 2))
        self.time_windows[0] = [0, 1000]  # Depot always open
        for i in range(1, self.n_nodes):
            start = 0  # Can arrive anytime after 0
            width = 800  # Wide window for feasibility
            self.time_windows[i] = [start, start + width]

        print(f"✓ Generated time windows for all customers")

        # =====================================================================
        # STEP 1.6: Service Times
        # =====================================================================
        # How long it takes to serve each customer (unloading, paperwork, etc.)
        self.service_times = np.ones(self.n_nodes) * 5
        self.service_times[0] = 0  # No service time at depot

        print(f"✓ Service time: {self.service_times[1]} time units per customer")

        # =====================================================================
        # STEP 1.7: Skills (IMPORTANT FOR YOUR PROBLEM!)
        # =====================================================================
        # Define 3 types of skills (you can customize these):
        # Skill 0: Heavy lifting capability
        # Skill 1: Refrigeration equipment
        # Skill 2: Fragile item handling

        # What skills each customer requires
        self.customer_skills = []
        for i in range(self.n_nodes):
            if i == 0:
                self.customer_skills.append(set())  # Depot needs no skills
            else:
                # Each customer needs 0 or 1 skill
                n_skills = np.random.randint(0, 2)
                if n_skills > 0:
                    skills = set([np.random.choice(3)])
                else:
                    skills = set()
                self.customer_skills.append(skills)

        # What skills each vehicle has
        self.vehicle_skills = []
        for v in range(n_vehicles):
            # Each vehicle has 2-3 skills
            n_skills = np.random.randint(2, 4)
            skills = set(np.random.choice(3, n_skills, replace=False))
            self.vehicle_skills.append(skills)

        print(f"✓ Generated skills for customers and vehicles")
        print(f"  Vehicle skills: {[sorted(s) for s in self.vehicle_skills]}")

    # =========================================================================
    # STEP 2: BUILD AND SOLVE THE OPTIMIZATION MODEL
    # =========================================================================
    def solve(self):
        """
        Build the mathematical model and solve it using Gurobi
        """

        # Create optimization model
        model = gp.Model("VRP_TimeWindows_Skills")
        model.setParam('OutputFlag', 1)  # Show Gurobi output
        model.setParam('TimeLimit', 300)  # Max 5 minutes
        model.setParam('MIPFocus', 1)  # Focus on finding feasible solutions
        model.setParam('Heuristics', 0.3)  # Spend 30% time on heuristics

        print("\n" + "=" * 70)
        print("BUILDING OPTIMIZATION MODEL")
        print("=" * 70)

        # =====================================================================
        # STEP 2.1: DECISION VARIABLES
        # =====================================================================
        print("\n[Step 2.1] Creating decision variables...")

        # x[i,j,v] = 1 if vehicle v travels from location i to location j
        #          = 0 otherwise
        # This is a BINARY variable (0 or 1)
        x = {}
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j:  # Can't travel from a node to itself
                    for v in range(self.n_vehicles):
                        x[i, j, v] = model.addVar(
                            vtype=GRB.BINARY,
                            name=f'x_{i}_{j}_{v}'
                        )

        print(f"  ✓ Created {len(x)} binary routing variables")

        # t[i,v] = time when vehicle v arrives at location i
        # This is a CONTINUOUS variable (can be any number within bounds)
        t = {}
        for i in range(self.n_nodes):
            for v in range(self.n_vehicles):
                t[i, v] = model.addVar(
                    vtype=GRB.CONTINUOUS,
                    lb=self.time_windows[i][0],  # Lower bound = earliest time
                    ub=self.time_windows[i][1],  # Upper bound = latest time
                    name=f't_{i}_{v}'
                )

        print(f"  ✓ Created {len(t)} continuous time variables")

        # =====================================================================
        # STEP 2.2: OBJECTIVE FUNCTION
        # =====================================================================
        print("\n[Step 2.2] Setting objective function...")

        # MINIMIZE: Total distance traveled by all vehicles
        # Sum of (distance * whether that arc is used)
        obj = gp.quicksum(
            self.dist[i][j] * x[i, j, v]
            for i in range(self.n_nodes)
            for j in range(self.n_nodes)
            if i != j
            for v in range(self.n_vehicles)
        )
        model.setObjective(obj, GRB.MINIMIZE)

        print(f"  ✓ Objective: Minimize total travel distance")

        # =====================================================================
        # STEP 2.3: CONSTRAINT 1 - Visit Each Customer Exactly Once
        # =====================================================================
        print("\n[Step 2.3] Adding constraint: Each customer visited once...")

        # For each customer j (excluding depot):
        # Sum of all incoming arcs to j (from any node i, by any vehicle v) = 1
        for j in range(1, self.n_nodes):
            model.addConstr(
                gp.quicksum(x[i, j, v]
                            for i in range(self.n_nodes) if i != j
                            for v in range(self.n_vehicles)) == 1,
                name=f'visit_{j}'
            )

        print(f"  ✓ Added {self.n_customers} visit constraints")

        # =====================================================================
        # STEP 2.4: CONSTRAINT 2 - Flow Conservation
        # =====================================================================
        print("\n[Step 2.4] Adding constraint: Flow conservation...")

        # For each vehicle v and each node j:
        # If a vehicle enters node j, it must also leave node j
        # (number of incoming arcs = number of outgoing arcs)
        for v in range(self.n_vehicles):
            for j in range(self.n_nodes):
                model.addConstr(
                    gp.quicksum(x[i, j, v] for i in range(self.n_nodes) if i != j) ==
                    gp.quicksum(x[j, i, v] for i in range(self.n_nodes) if i != j),
                    name=f'flow_{j}_{v}'
                )

        print(f"  ✓ Added {self.n_vehicles * self.n_nodes} flow constraints")

        # =====================================================================
        # STEP 2.5: CONSTRAINT 3 & 4 - Start and End at Depot
        # =====================================================================
        print("\n[Step 2.5] Adding constraint: Routes start/end at depot...")

        # Each vehicle can leave depot at most once
        for v in range(self.n_vehicles):
            model.addConstr(
                gp.quicksum(x[0, j, v] for j in range(1, self.n_nodes)) <= 1,
                name=f'start_depot_{v}'
            )

        # Each vehicle can return to depot at most once
        for v in range(self.n_vehicles):
            model.addConstr(
                gp.quicksum(x[i, 0, v] for i in range(1, self.n_nodes)) <= 1,
                name=f'end_depot_{v}'
            )

        print(f"  ✓ Added {2 * self.n_vehicles} depot constraints")

        # =====================================================================
        # STEP 2.6: CONSTRAINT 5 - Vehicle Capacity
        # =====================================================================
        print("\n[Step 2.6] Adding constraint: Vehicle capacity...")

        # For each vehicle v:
        # Sum of demands of all customers visited ≤ vehicle capacity
        for v in range(self.n_vehicles):
            model.addConstr(
                gp.quicksum(self.demands[j] * x[i, j, v]
                            for i in range(self.n_nodes)
                            for j in range(1, self.n_nodes)
                            if i != j) <= self.capacities[v],
                name=f'capacity_{v}'
            )

        print(f"  ✓ Added {self.n_vehicles} capacity constraints")

        # =====================================================================
        # STEP 2.7: CONSTRAINT 6 - Time Windows
        # =====================================================================
        print("\n[Step 2.7] Adding constraint: Time windows...")

        # If vehicle v travels from i to j:
        # arrival_time[j] ≥ arrival_time[i] + service_time[i] + travel_time[i,j]
        #
        # We use "Big M" method to only enforce this when x[i,j,v] = 1
        M = 2000  # Large constant

        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j:
                    for v in range(self.n_vehicles):
                        # If x[i,j,v] = 1: t[j,v] ≥ t[i,v] + service + distance
                        # If x[i,j,v] = 0: constraint becomes inactive (due to M)
                        model.addConstr(
                            t[i, v] + self.service_times[i] + self.dist[i][j]
                            <= t[j, v] + M * (1 - x[i, j, v]),
                            name=f'time_{i}_{j}_{v}'
                        )

        print(f"  ✓ Added time window constraints")

        # =====================================================================
        # STEP 2.8: CONSTRAINT 7 - Skill Matching (KEY FOR YOUR PROBLEM!)
        # =====================================================================
        print("\n[Step 2.8] Adding constraint: Skill matching...")

        # A vehicle can only visit a customer if the vehicle has
        # ALL the skills required by that customer

        skill_constraints_added = 0
        for j in range(1, self.n_nodes):
            required_skills = self.customer_skills[j]
            for v in range(self.n_vehicles):
                vehicle_skills = self.vehicle_skills[v]

                # Check if vehicle has all required skills
                if not required_skills.issubset(vehicle_skills):
                    # Vehicle v cannot visit customer j
                    # Set x[i,j,v] = 0 for all i
                    for i in range(self.n_nodes):
                        if i != j:
                            model.addConstr(
                                x[i, j, v] == 0,
                                name=f'skill_{i}_{j}_{v}'
                            )
                            skill_constraints_added += 1

        print(f"  ✓ Added {skill_constraints_added} skill matching constraints")

        # =====================================================================
        # STEP 2.9: SOLVE THE MODEL
        # =====================================================================
        print("\n" + "=" * 70)
        print("SOLVING MODEL WITH GUROBI...")
        print("=" * 70 + "\n")

        model.optimize()

        # =====================================================================
        # STEP 2.10: EXTRACT SOLUTION
        # =====================================================================
        print("\n" + "=" * 70)
        print("EXTRACTING SOLUTION...")
        print("=" * 70)

        # Check if we have a solution
        if model.status == GRB.OPTIMAL:
            print("✅ OPTIMAL SOLUTION FOUND!")
        elif model.status == GRB.TIME_LIMIT and model.SolCount > 0:
            print("⚠️  TIME LIMIT REACHED - Using best solution found")
        else:
            print(f"❌ NO SOLUTION FOUND (Status code: {model.status})")
            if model.status == GRB.TIME_LIMIT:
                print("   Reason: Time limit reached without finding feasible solution")
            elif model.status == GRB.INFEASIBLE:
                print("   Reason: Problem is infeasible (constraints too tight)")
            elif model.status == GRB.UNBOUNDED:
                print("   Reason: Problem is unbounded")
            return None, None, None

        # Extract solution only if we have one
        if model.SolCount > 0:
            self.routes = [[] for _ in range(self.n_vehicles)]
            self.times = [[] for _ in range(self.n_vehicles)]

            # For each vehicle, reconstruct its route
            for v in range(self.n_vehicles):
                current = 0  # Start at depot
                route = [0]
                route_times = [t[0, v].X]

                # Follow the route by finding x[current,j,v] = 1
                while True:
                    next_node = None
                    for j in range(self.n_nodes):
                        if j != current and x[current, j, v].X > 0.5:  # = 1
                            next_node = j
                            break

                    if next_node is None or next_node == 0:
                        if next_node == 0:
                            route.append(0)
                            route_times.append(t[0, v].X)
                        break

                    route.append(next_node)
                    route_times.append(t[next_node, v].X)
                    current = next_node

                if len(route) > 1:  # Vehicle was used
                    self.routes[v] = route
                    self.times[v] = route_times

            return model.objVal, self.routes, self.times
        else:
            return None, None, None

    # =========================================================================
    # STEP 3: DISPLAY THE SOLUTION
    # =========================================================================
    def print_solution(self, obj_val, routes, times):
        """
        Print the solution in a readable format
        """
        print(f"\n{'=' * 60}")
        print(f"SOLUTION SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total Distance: {obj_val:.2f} units")
        print(f"\nVehicle Skills:")
        for v in range(self.n_vehicles):
            print(f"  Vehicle {v}: Skills {sorted(self.vehicle_skills[v])}")

        print(f"\n{'=' * 60}")
        for v in range(self.n_vehicles):
            if len(routes[v]) > 1:
                print(f"\n🚚 VEHICLE {v} ROUTE:")
                print(f"   Capacity: {self.capacities[v]} units")
                print(f"   Skills: {sorted(self.vehicle_skills[v])}")
                print(f"   {'-' * 55}")

                total_demand = 0
                total_distance = 0

                for idx in range(len(routes[v])):
                    node = routes[v][idx]
                    time = times[v][idx] if idx < len(times[v]) else 0

                    if node == 0:
                        print(f"   [{idx}] 🏢 DEPOT (arrival time: {time:.1f})")
                    else:
                        total_demand += self.demands[node]
                        req_skills = sorted(self.customer_skills[node])
                        tw = self.time_windows[node]
                        print(f"   [{idx}] 📦 Customer {node}:")
                        print(f"       • Demand: {self.demands[node]} units")
                        print(f"       • Arrival time: {time:.1f}")
                        print(f"       • Time window: [{tw[0]:.0f}, {tw[1]:.0f}]")
                        print(f"       • Required skills: {req_skills}")

                    if idx < len(routes[v]) - 1:
                        dist = self.dist[routes[v][idx]][routes[v][idx + 1]]
                        total_distance += dist
                        print(f"       ↓ Travel {dist:.1f} units")

                print(f"   {'-' * 55}")
                print(f"   📊 Total Load: {total_demand}/{self.capacities[v]} units")
                print(f"   📏 Total Distance: {total_distance:.2f} units")

    def plot_solution(self, routes):
        """
        Visualize the solution on a 2D map
        """
        plt.figure(figsize=(12, 10))

        # Plot depot
        plt.scatter(self.locations[0][0], self.locations[0][1],
                    c='red', s=300, marker='s', label='Depot', zorder=3)
        plt.text(self.locations[0][0], self.locations[0][1] + 3,
                 'DEPOT', ha='center', fontweight='bold', fontsize=10)

        # Plot customers
        for i in range(1, self.n_nodes):
            skills_str = ','.join(map(str, sorted(self.customer_skills[i])))
            if not skills_str:
                skills_str = 'none'
            plt.scatter(self.locations[i][0], self.locations[i][1],
                        c='blue', s=100, zorder=2)
            plt.text(self.locations[i][0], self.locations[i][1] + 2,
                     f'{i}\nSkills:[{skills_str}]', ha='center', fontsize=7)

        # Plot routes
        colors = plt.cm.rainbow(np.linspace(0, 1, self.n_vehicles))
        for v, route in enumerate(routes):
            if len(route) > 1:
                for i in range(len(route) - 1):
                    start = self.locations[route[i]]
                    end = self.locations[route[i + 1]]
                    plt.arrow(start[0], start[1],
                              end[0] - start[0], end[1] - start[1],
                              color=colors[v], width=0.3,
                              head_width=2, length_includes_head=True,
                              label=f'Vehicle {v} (Skills: {sorted(self.vehicle_skills[v])})' if i == 0 else '',
                              alpha=0.7, zorder=1)

        plt.legend(loc='upper right', fontsize=9)
        plt.title('VRP Solution: Delivery Routes with Time Windows & Skills',
                  fontsize=14, fontweight='bold')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("VEHICLE ROUTING PROBLEM WITH TIME WINDOWS AND SKILLS")
    print("=" * 70)

    # Create problem instance
    print("\n[INITIALIZATION] Creating problem instance...")
    vrp = VRPTimWindowsSkills(n_customers=10, n_vehicles=3)  # Reduced from 15 to 10

    print(f"\nProblem Configuration:")
    print(f"  • Customers: {vrp.n_customers}")
    print(f"  • Vehicles: {vrp.n_vehicles}")
    print(f"  • Total locations: {vrp.n_nodes} (including depot)")

    # Solve
    obj_val, routes, times = vrp.solve()

    # Print solution
    if obj_val is not None:
        vrp.print_solution(obj_val, routes, times)
        vrp.plot_solution(routes)
    else:
        print("\n" + "=" * 70)
        print("❌ COULD NOT FIND A FEASIBLE SOLUTION")
        print("=" * 70)
        print("\n🔍 DIAGNOSTICS:")
        print(f"   • Total demand: {sum(vrp.demands[1:])} units")
        print(f"   • Total capacity: {sum(vrp.capacities)} units")
        print(f"   • Capacity utilization needed: {sum(vrp.demands[1:]) / sum(vrp.capacities) * 100:.1f}%")

        # Check skill coverage
        print("\n   • Skill requirements check:")
        for i in range(1, vrp.n_nodes):
            req_skills = vrp.customer_skills[i]
            can_serve = []
            for v in range(vrp.n_vehicles):
                if req_skills.issubset(vrp.vehicle_skills[v]):
                    can_serve.append(v)
            if not can_serve:
                print(f"     ⚠️  Customer {i} (needs {sorted(req_skills)}) - NO vehicle can serve!")
            elif len(can_serve) == 1:
                print(f"     ⚠️  Customer {i} (needs {sorted(req_skills)}) - Only vehicle {can_serve[0]} can serve")

        print("\n💡 SUGGESTIONS:")
        print("   1. Reduce number of customers")
        print("   2. Increase vehicle capacities")
        print("   3. Ensure vehicles have broader skill sets")
        print("   4. Relax time windows further")
        print("   5. Add more vehicles")