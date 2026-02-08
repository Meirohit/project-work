from Problem import Problem
import networkx as nx
import random
import copy
import random
import networkx as nx
#from src.helper_functions import create_neighbor
from src.ga_algorithm import genetic_algorithm


def solution(p: Problem):
    """
    Solve TTP using Genetic Algorithm
    """
    print("=" * 60)
    print("GENETIC ALGORITHM SOLUTION")
    print("=" * 60)
    
    best_solution = genetic_algorithm(
        problem=p,
        population_size=100,
        generations=200,
        crossover_rate=0.7,
        mutation_rate=0.3,
        tournament_size=3,
        elite_size=3,
        verbose=True
    )
    
    return best_solution.path_steps


if __name__ == "__main__":
    # test_configs = [
    #     {"population_size": 50, "generations": 100, "mutation_rate": 0.1},
    #     {"population_size": 100, "generations": 200, "mutation_rate": 0.2},
    #     {"population_size": 150, "generations": 150, "mutation_rate": 0.3},
    # ]
    
    # p = Problem(num_cities=100, density=0.2, alpha=1.0, beta=1.0)
    # baseline = p.baseline()
    
    # print(f"Baseline: {baseline:.2f}\n")
    
    # for config in test_configs:
    #     print(f"\nTesting config: {config}")
    #     best = genetic_algorithm(p, **config, verbose=False)
    #     print(f"Result: {-best.fitness:.2f}")

    p = Problem(num_cities=200, density=1, alpha=2, beta=0.6, seed=42)
    
    print(f"Baseline cost: {p.baseline():.2f}\n")
    
    result_path = solution(p)
    
    print(f"\nFinal path length: {len(result_path)}")
    print(f"First 10 steps: {result_path[:10]}")
    