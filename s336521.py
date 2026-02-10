from Problem import Problem
import networkx as nx
import random
import copy
import random
import networkx as nx
#from src.helper_functions import create_neighbor
from src.hybrid_aco.hybrid_algorithm import fast_hybrid_aco_ttp
from src.aco_algorithm import ant_colony_optimization
from src.ga_algorithm import genetic_algorithm
import time


def solution(p: Problem):
    """
    Solve TTP using ACO or GA or Hybrid ACO
    """
    
    # Choose algorithm
    algorithm = "GA"     # Options: "ACO", "GA", "HYBRID"
    
    if algorithm == "ACO":
        print("=" * 60)
        print("Ant Colony Optimization SOLUTION")
        print("=" * 60)
        best_solution = ant_colony_optimization(
            problem=p,
            num_ants=50,
            num_iterations=100,
            alpha=1.0,      # Pheromone importance
            beta=2.5,       # Heuristic importance (higher = more greedy)
            rho=0.1,        # Evaporation rate
            Q=100,
            elite_weight=2.0,
            verbose=True
        )
        return best_solution.path_steps
    
    elif algorithm == "GA":
        print("=" * 60)
        print("GENETIC ALGORITHM SOLUTION")
        print("=" * 60)
        
        best_solution = genetic_algorithm(
            problem=p,
            population_size=100,
            generations=200,
            crossover_rate=0.8,
            mutation_rate=0.2,
            tournament_size=3,
            elite_size=3,
            verbose=True
        )
        
        return best_solution.path_steps
    
    elif algorithm == "HYBRID":
        print("=" * 60)
        print("HYBRID ACO SOLUTION")
        print("=" * 60)
        
        """
        Fast optimized solution
        """
        
        path_steps, final_cost = fast_hybrid_aco_ttp(
            problem=p,
            num_ants=50,           
            num_iterations=80,     
            alpha=1.0,             # Pheromone importance
            beta=2.5,              # Heuristic importance (higher = more greedy)
            q0=0.8,                # More exploitation
            rho_global=0.1,         
            inver_over_prob=0.7,  
            optimize_trips=True,
            verbose=True
        )
        
        return path_steps


if __name__ == "__main__":
    p = Problem(num_cities=100, density=1, alpha=1, beta=3, seed=42)
    baseline = p.baseline()
    print(f"Baseline: {baseline:.2f}\n")
    
    ########################################## GA TESTS #####################################################
    # test_configs = [
    #     {"population_size": 50, "generations": 100, "mutation_rate": 0.1},
    #     {"population_size": 100, "generations": 200, "mutation_rate": 0.2},
    #     {"population_size": 150, "generations": 150, "mutation_rate": 0.3},
    # ]
    # for config in test_configs:
    #     print(f"\nTesting config: {config}")
    #     best = genetic_algorithm(p, **config, verbose=False)
    #     print(f"Result: {-best.fitness:.2f}")
    #
    # Test one GA
    print("\n" + "="*60)
    print("TESTING GA")
    print("="*60)
    result = solution(p)                      # set "GA" in solution() to test GA

    # # Validation check
    # if result_ga.path_steps:
    #     trips = 0
    #     for node, _ in result_ga.path_steps:
    #         if node == 0: trips += 1
    #     # Subtract 1 because trips usually end with 0, but we count returns
    #     print(f"Solution strategy used approx {(trips-1)//2 + 1} trips (returns to depot).")
    ##########################################################################################################
    

    ########################################## HYBRID ACO TESTS ##############################################
    # print("\n" + "="*60)
    # print("TESTING HYBRID ACO")
    # print("="*60)
    # start = time.time()
    # result = solution(p)                      # set "HYBRID" in solution() to test hybrid ACO
    # elapsed = time.time() - start
    
    # print(f"\nTotal time: {elapsed:.2f} seconds")
    # print(f"Path length: {len(result)} steps")
    
    # test_configs = [
    #     (30, 0.3, 1.0, 1.0),  # Small, moderate beta
    #     (50, 0.3, 1.0, 0.5),  # Medium, high beta
    #     (100, 0.2, 1.5, 2.5),  # Large, very high beta
    # ]
    # for n, d, a, b in test_configs:
    #     print("\n" + "="*70)
    #     print(f"TEST: {n} cities, density={d}, alpha={a}, beta={b}")
    #     print("="*70)
    #     p = Problem(num_cities=n, density=d, alpha=a, beta=b, seed=42)
    #     baseline = p.baseline()
    #     print(f"\nBaseline cost: {baseline:.2f}\n")
    #     result = solution(p)                  # set "HYBRID" in solution() to test hybrid ACO
    #     print(f"\nPath length: {len(result)} steps")
        
    #     # Verify all gold collected
    #     gold_in_cities = {i: p.graph.nodes[i]['gold'] for i in range(1, n)}
    #     gold_collected = {}
        
    #     for city, gold in result:
    #         if city != 0:
    #             gold_collected[city] = gold_collected.get(city, 0) + gold
        
    #     print("\nVerification:")
    #     all_collected = all(
    #         abs(gold_collected.get(city, 0) - gold_in_cities[city]) < 0.01
    #         for city in gold_in_cities
    #     )
    #     print(f"All gold collected: {all_collected}")

    ##########################################################################################################


    ########################################## SIMPLE ACO TESTS ##############################################
    # print("\n" + "="*60)
    # print("TESTING SIMPLE ACO")
    # print("="*60)
    # result_aco = ant_colony_optimization(p, num_ants=40, num_iterations=80, verbose=True, alpha=3, beta=1.5)
    # print(f"ACO Result: {result_aco.total_cost:.2f}")
    #
    # Test different alpha/beta combinations
    # configs = [
    #     {"alpha": 1.0, "beta": 2.0},  # Balanced
    #     {"alpha": 0.5, "beta": 3.0},  # More heuristic (greedy)
    #     {"alpha": 2.0, "beta": 1.0},  # More pheromone (exploration)
    # ]
    # for config in configs:
    #     result = ant_colony_optimization(p, **config, num_ants=30, num_iterations=50)
    #     print(f"Alpha={config['alpha']}, Beta={config['beta']}: Cost={result.total_cost:.2f}")
    ##########################################################################################################


    