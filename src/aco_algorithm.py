from Problem import Problem
from src.aco_pheromone import PheromoneMatrix
from src.aco_ant import Ant, evaluate_aco_solution
import random

def ant_colony_optimization(
    problem: Problem,
    num_ants=50,
    num_iterations=100,
    alpha=1.0,           
    beta=2.0,            
    rho=0.1,             
    Q=100,               # Pheromone deposit factor
    elite_weight=2.0,    # Extra pheromone for best solution
    verbose=True
):
    """
    Ant Colony Optimization for TTP
    """
    
    num_cities = len(problem.graph.nodes)
    
    # Initialize pheromone matrix
    pheromone = PheromoneMatrix(num_cities, initial_pheromone=1.0)
    
    # Track best solution found
    best_solution = None
    best_cost = float('inf')
    
    if verbose:
        print("=" * 60)
        print("ANT COLONY OPTIMIZATION")
        print("=" * 60)
        print(f"Ants: {num_ants}, Iterations: {num_iterations}")
        print(f"Alpha: {alpha}, Beta: {beta}, Rho: {rho}")
        print()
    
    # Main ACO loop
    for iteration in range(num_iterations):
        # Store all solutions from this iteration
        iteration_solutions = []
        
        # Each ant constructs a solution
        for ant_id in range(num_ants):
            # Create ant
            ant = Ant(problem, pheromone, alpha=alpha, beta=beta)
            
            # Ant builds solution
            solution = ant.construct_solution()
            
            # Evaluate solution
            cost = evaluate_aco_solution(solution, problem)
            
            iteration_solutions.append(solution)
            
            # Update best solution
            if cost < best_cost:
                best_cost = cost
                best_solution = solution
                
                if verbose:
                    print(f"Iteration {iteration}, Ant {ant_id}: New best cost = {best_cost:.2f}")
        
        # Pheromone evaporation
        pheromone.evaporate(rho)
        
        # Pheromone deposit
        for solution in iteration_solutions:
            # Amount of pheromone to deposit (inversely proportional to cost)
            # Better solutions (lower cost) deposit MORE pheromone
            deposit_amount = Q / solution.total_cost
            
            # Deposit pheromone along the route
            pheromone.deposit(solution.visited_order, deposit_amount)
        
        # Elite strategy: best solution deposits extra pheromone
        if best_solution:
            elite_amount = (Q / best_solution.total_cost) * elite_weight
            pheromone.deposit(best_solution.visited_order, elite_amount)
        
        # Progress report
        if verbose and iteration % 10 == 0:
            avg_cost = sum(s.total_cost for s in iteration_solutions) / len(iteration_solutions)
            print(f"Iteration {iteration}: Best={best_cost:.2f}, Avg={avg_cost:.2f}")
    
    if verbose:
        print()
        print(f"Final best cost: {best_cost:.2f}")
        print("=" * 60)
    
    return best_solution