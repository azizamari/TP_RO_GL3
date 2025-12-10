import sys

PROBLEMS = {
    '1': {'name': 'Problème 4.5 - Localisation-Allocation (Centres de Tri)', 'module': 'problem_location_allocation'},
    '2': {'name': 'Problème 2 - Tarification Optimale de l\'Électricité', 'module': 'problem_2'},
    '3': {'name': 'Problème 11.4 - Routage du personnel ', 'module': 'problem_11_4_routage'},
    '4': {'name': 'Problème 9.4 - Sélection Investissements (Énergie)', 'module': 'problem_9_4_energy'},
    '5': {'name': 'Problème 5 - [À définir]', 'module': 'problem_5'},
}

def print_header():
    print("=" * 80)
    print(" " * 20 + "INTERFACE D'OPTIMISATION - ÉQUIPE GL3")
    print("=" * 80)
    print()

def print_menu():
    print("\nSélectionnez un problème à résoudre:")
    print("-" * 80)
    for key, problem in PROBLEMS.items():
        print(f"{key}. {problem['name']}")
    print("0. Quitter")
    print("-" * 80)

def main():
    while True:
        print_header()
        print_menu()
        
        try:
            choix = input(f"\nVotre choix (0-{len(PROBLEMS)}): ").strip()
            
            if choix == '0':
                print("\nAu revoir!")
                break
            elif choix in PROBLEMS:
                problem = PROBLEMS[choix]
                print(f"\n[{problem['name']}]")
                try:
                    module = __import__(problem['module'])
                    module.solve()
                except ImportError:
                    print(f"Module {problem['module']}.py non trouvé. Utilisez problem_template.py.")
                except Exception as e:
                    print(f"Erreur: {e}")
            else:
                print(f"\nChoix invalide! Entrez un nombre entre 0 et {len(PROBLEMS)}.")
            
            input("\nAppuyez sur Entrée pour continuer...")
            
        except KeyboardInterrupt:
            print("\n\nInterruption détectée. Au revoir!")
            break
        except Exception as e:
            print(f"\nErreur inattendue: {e}")
            input("\nAppuyez sur Entrée pour continuer...")

if __name__ == "__main__":
    main()
