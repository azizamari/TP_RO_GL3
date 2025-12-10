from gurobipy import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import gradio as gr


class LocationAllocationSolver:
    """Solveur avancé pour le problème de localisation-allocation"""
    
    def __init__(self):
        # Données du problème - Enrichies avec plus d'attributs
        self.quartiers = {
            'noms': [
                "Centre-Ville", 
                "Zone Industrielle Nord", 
                "Quartier Résidentiel Est",
                "Banlieue Sud", 
                "Zone Commerciale Ouest", 
                "Quartier Universitaire",
                "Zone Portuaire", 
                "Quartier Historique"
            ],
            # Demande par type de déchet (recyclable, organique, dangereux) en tonnes/semaine
            'demande_recyclable': [80, 150, 120, 100, 130, 70, 180, 50],
            'demande_organique': [50, 90, 60, 60, 70, 40, 90, 30],
            'demande_dangereux': [20, 40, 20, 20, 20, 10, 30, 10],
            # Coordonnées pour visualisation
            'x': [50, 30, 80, 50, 20, 60, 40, 55],
            'y': [50, 80, 50, 20, 50, 70, 30, 60]
        }
        
        self.sites = {
            'noms': [
                "Site A - Zone Nord", 
                "Site B - Zone Est", 
                "Site C - Zone Sud",
                "Site D - Zone Ouest", 
                "Site E - Centre"
            ],
            # Capacités par type de déchet
            'capacite_recyclable': [350, 300, 400, 320, 250],
            'capacite_organique': [200, 150, 250, 180, 150],
            'capacite_dangereux': [50, 50, 50, 50, 50],
            # Capacité volumique (m³/semaine)
            'capacite_volume': [800, 700, 900, 750, 600],
            'cout_fixe': [250, 200, 280, 220, 300],
            # Émissions CO2 (tonnes/an)
            'emissions_co2': [120, 100, 140, 110, 90],
            # Fenêtres de temps disponibles
            'fenetres_disponibles': [[1,2], [2,3], [1,2,3], [1,3], [1,2,3]],
            # Coordonnées
            'x': [35, 75, 45, 25, 50],
            'y': [75, 55, 25, 45, 50]
        }
        
        # Matrice de coûts de transport (TND/tonne)
        self.cout_transport = np.array([
            [15, 25, 35, 30, 10],
            [12, 30, 40, 35, 25],
            [30, 10, 25, 40, 20],
            [35, 30, 12, 25, 30],
            [30, 40, 28, 10, 25],
            [20, 18, 28, 32, 15],
            [25, 35, 30, 22, 28],
            [18, 28, 38, 35, 12]
        ])
        
        # Facteur volumique (m³/tonne) par type de déchet
        self.facteur_volume = {
            'recyclable': 1.5,
            'organique': 1.0,
            'dangereux': 0.8
        }
        
        
        self.n_quartiers = len(self.quartiers['noms'])
        self.n_sites = len(self.sites['noms'])
        
    
    def solve(self, budget_max=800, max_sites=3, capacite_mult=1.0, 
              poids_co2=0.3,
              distance_max_penalite=40):
        """
        Résout le problème avec contraintes enrichies
        """
        
        model = Model("Location_Allocation_Avance")
        
        # ===== VARIABLES DE DÉCISION =====
        
        # y[j] = 1 si site j est ouvert
        y = {}
        for j in range(self.n_sites):
            y[j] = model.addVar(vtype=GRB.BINARY, name=f"Ouvrir_{j}")
        
        # proportion de la demande du quartier i, servie par site j
        x = {}
        for i in range(self.n_quartiers):
            for j in range(self.n_sites):
                x[i,j] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1,
                                        name=f"Alloc_{i}_{j}")
        
        # z[i,j] = 1 si pénalité de distance appliquée
        z = {}
        for i in range(self.n_quartiers):
            for j in range(self.n_sites):
                z[i,j] = model.addVar(vtype=GRB.BINARY, name=f"Penalite_{i}_{j}")
        
        
        # ===== FONCTION OBJECTIF MULTI-CRITÈRES =====
        
        # Objectif 1: Minimiser coûts (fixes + transport + pénalités)
        cout_fixe_total = quicksum(self.sites['cout_fixe'][j] * y[j] 
                                   for j in range(self.n_sites))
        
        demande_totale = [sum(self.quartiers[f'demande_{t}'][i] 
                                for t in ['recyclable', 'organique', 'dangereux'])
                            for i in range(self.n_quartiers)]
        cout_transport_total = quicksum(
            self.cout_transport[i,j] * demande_totale[i] * x[i,j]
            for i in range(self.n_quartiers)
            for j in range(self.n_sites)
        ) / 1000
        
        cout_penalite = quicksum(
            20 * z[i,j] * demande_totale[i] * x[i,j]
            for i in range(self.n_quartiers)
            for j in range(self.n_sites)
        ) / 1000
        
        cout_total = cout_fixe_total + cout_transport_total + cout_penalite
        
        # Objectif 2: Minimiser émissions CO2
        emissions_totales = quicksum(self.sites['emissions_co2'][j] * y[j] 
                                     for j in range(self.n_sites))
        
        # Objectif combiné avec pondération
        objectif = (1 - poids_co2) * cout_total + poids_co2 * emissions_totales / 100
        
        model.setObjective(objectif, GRB.MINIMIZE)
        
        
        # ===== CONTRAINTES =====
            # 1. Chaque quartier doit être entièrement desservi
        demande_totale = [sum(self.quartiers[f'demande_{t}'][i] 
                                for t in ['recyclable', 'organique', 'dangereux'])
                            for i in range(self.n_quartiers)]
        
            # 1. Chaque quartier doit être entièrement desservi
        for i in range(self.n_quartiers):
            model.addConstr(
                quicksum(x[i,j] for j in range(self.n_sites)) == 1,
                name=f"Desserte_{i}"
            )
        
            # 2. Un quartier ne peut être servi que par un site ouvert
        for i in range(self.n_quartiers):
            for j in range(self.n_sites):
                model.addConstr(x[i,j] <= y[j], name=f"Lien_{i}_{j}")
        
        # 3. Capacités des sites (poids et volume)
        capacite_totale = [sum(self.sites[f'capacite_{t}'][j] 
                                for t in ['recyclable', 'organique', 'dangereux'])
                            for j in range(self.n_sites)]
        

        for j in range(self.n_sites):
            model.addConstr(
                quicksum(demande_totale[i] * x[i,j] 
                        for i in range(self.n_quartiers)) <= 
                capacite_totale[j] * capacite_mult,
                name=f"Cap_{j}"
            )
        
        
        # 7. Pénalité de distance
        for i in range(self.n_quartiers):
            for j in range(self.n_sites):
                utilisation = x[i,j]
                if self.cout_transport[i,j] > distance_max_penalite:
                    model.addConstr(z[i,j] >= utilisation, name=f"Dist_{i}_{j}")
                else:
                    model.addConstr(z[i,j] == 0, name=f"NoDist_{i}_{j}")
        
        # 8. Budget
        model.addConstr(cout_fixe_total <= budget_max, name="Budget")
        
        # 9. Nombre max de sites
        model.addConstr(
            quicksum(y[j] for j in range(self.n_sites)) <= max_sites,
            name="Max_Sites"
        )
        
        # ===== RÉSOLUTION =====
        
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', 300)
        model.setParam('MIPGap', 0.01)
        model.optimize()
        
        
        # ===== EXTRACTION DES RÉSULTATS =====
        
        resultats = {
            'status': model.status,
            'optimal': model.status == GRB.OPTIMAL,
            'infaisable': model.status == GRB.INFEASIBLE
        }
        
        if model.status == GRB.OPTIMAL:
            # Calcul des coûts réels séparément
            cout_fixe_reel = sum(self.sites['cout_fixe'][j] * y[j].x 
                                for j in range(self.n_sites))
            
            demande_totale = [sum(self.quartiers[f'demande_{t}'][i] 
                                    for t in ['recyclable', 'organique', 'dangereux'])
                                for i in range(self.n_quartiers)]
            cout_transport_reel = sum(
                self.cout_transport[i,j] * demande_totale[i] * x[i,j].x
                for i in range(self.n_quartiers)
                for j in range(self.n_sites)
            ) / 1000
            
            cout_penalite_reel = sum(
                20 * z[i,j].x * demande_totale[i] * x[i,j].x
                for i in range(self.n_quartiers)
                for j in range(self.n_sites)
            ) / 1000
            
            emissions_reel = sum(self.sites['emissions_co2'][j] * y[j].x 
                                for j in range(self.n_sites))
            
            resultats.update({
                'objectif_gurobi': model.objVal,
                'cout_fixe': cout_fixe_reel,
                'cout_transport': cout_transport_reel,
                'cout_penalite': cout_penalite_reel,
                'cout_total_reel': cout_fixe_reel + cout_transport_reel + cout_penalite_reel,
                'emissions_co2': emissions_reel,
                'sites_ouverts': [j for j in range(self.n_sites) if y[j].x > 0.5],
                'affectations': [],
                'charges_sites': {},
                'statistiques': {}
            })
            
            # Calcul des charges par site
            for j in resultats['sites_ouverts']:
                charges = {}
                charge = sum(demande_totale[i] * x[i,j].x 
                            for i in range(self.n_quartiers))
                charges['total'] = charge
                
                resultats['charges_sites'][j] = charges
            
            # Affectations détaillées
            for i in range(self.n_quartiers):
                for j in range(self.n_sites):
                    if x[i,j].x > 0.01:
                        resultats['affectations'].append({
                            'quartier': self.quartiers['noms'][i],
                            'site': self.sites['noms'][j],
                            'type_dechet': 'tous',
                            'proportion': x[i,j].x * 100,
                            'demande': demande_totale[i] * x[i,j].x,
                            'cout_transport': self.cout_transport[i,j]
                        })
            
            # Statistiques globales
            demande_totale_globale = sum(
                sum(self.quartiers[f'demande_{t}'][i] 
                    for t in ['recyclable', 'organique', 'dangereux'])
                for i in range(self.n_quartiers)
            )
            
            capacite_totale_installee = sum(
                sum(self.sites[f'capacite_{t}'][j] 
                    for t in ['recyclable', 'organique', 'dangereux'])
                for j in resultats['sites_ouverts']
            ) * capacite_mult
            
            resultats['statistiques'] = {
                'demande_totale': demande_totale_globale,
                'capacite_installee': capacite_totale_installee,
                'taux_utilisation': (demande_totale_globale / capacite_totale_installee) * 100,
                'nb_sites_ouverts': len(resultats['sites_ouverts']),
                'budget_utilise': cout_fixe_reel,
                'cout_transport_moyen': cout_transport_reel / demande_totale_globale * 1000 if demande_totale_globale > 0 else 0
            }
        
        elif model.status == GRB.INFEASIBLE:
            model.computeIIS()
            resultats['message'] = "Problème infaisable - Contraintes incompatibles"
        
        return resultats, model
    
    
    def visualiser_solution(self, resultats, save_path=None):
        """
        Crée des visualisations graphiques de la solution
        """
        if not resultats['optimal']:
            return None
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Carte des sites et affectations
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        ax1.set_title('Carte des Sites et Affectations', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Position X')
        ax1.set_ylabel('Position Y')
        ax1.grid(True, alpha=0.3)
        
        # Tracer les affectations (lignes)
        for aff in resultats['affectations']:
            i = self.quartiers['noms'].index(aff['quartier'])
            j_idx = next((idx for idx, nom in enumerate(self.sites['noms']) if nom == aff['site']), None)
            if j_idx is not None:
                ax1.plot([self.quartiers['x'][i], self.sites['x'][j_idx]],
                        [self.quartiers['y'][i], self.sites['y'][j_idx]],
                        'b-', alpha=0.2, linewidth=aff['proportion']/20)
        
        # Tracer les quartiers (cercles bleus)
        for i in range(self.n_quartiers):
            demande_tot = sum(self.quartiers[f'demande_{t}'][i] 
                             for t in ['recyclable', 'organique', 'dangereux'])
            ax1.scatter(self.quartiers['x'][i], self.quartiers['y'][i],
                       s=demande_tot*2, c='skyblue', edgecolors='blue',
                       linewidths=2, alpha=0.7, zorder=3)
            ax1.text(self.quartiers['x'][i], self.quartiers['y'][i]-3,
                    f"{demande_tot:.0f}t", ha='center', fontsize=8)
        
        # Tracer les sites ouverts (étoiles vertes)
        for j in resultats['sites_ouverts']:
            charge = resultats['charges_sites'][j]['total']
            ax1.scatter(self.sites['x'][j], self.sites['y'][j],
                       s=500, c='lightgreen', marker='*', edgecolors='darkgreen',
                       linewidths=3, zorder=4)
            ax1.text(self.sites['x'][j], self.sites['y'][j]+4,
                    f"Site {chr(65+j)}\n{charge:.0f}t", ha='center',
                    fontsize=9, fontweight='bold')
        
        
        
        # 3. Utilisation des sites (barres)
        ax3 = fig.add_subplot(gs[1, 2])
        ax3.set_title('Utilisation des Sites', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Taux d\'utilisation (%)')
        
        sites_noms = [f"Site {chr(65+j)}" for j in resultats['sites_ouverts']]
        utilisations = []
        
        for j in resultats['sites_ouverts']:
            charge = resultats['charges_sites'][j]['total']
            capacite = sum(self.sites[f'capacite_{t}'][j] 
                          for t in ['recyclable', 'organique', 'dangereux'])
            taux = (charge / capacite) * 100 if capacite > 0 else 0
            utilisations.append(taux)
        
        colors_bars = ['green' if u < 90 else 'orange' if u < 100 else 'red' 
                      for u in utilisations]
        
        ax3.bar(sites_noms, utilisations, color=colors_bars, alpha=0.7)
        ax3.axhline(y=90, color='orange', linestyle='--', label='Seuil optimal (90%)')
        ax3.axhline(y=100, color='red', linestyle='--', label='Capacité max')
        ax3.legend(fontsize=8)
        ax3.set_ylim(0, 120)
        
        
        plt.suptitle('Analyse Complète de la Solution d\'Optimisation',
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig