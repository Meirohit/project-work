import math
import random
import networkx as nx
from src.ga_solution import TTPSolution
from src.ga_evaluation import evaluate_solution_split, clear_path_cache
from src.ga_selection import tournament_selection, elitism_selection
from src.ga_operators import order_crossover, swap_mutation, inversion_mutation, insert_mutation
from src.hybrid_aco.precompute import PrecomputedData

def genetic_algorithm(
    problem,
    population_size=100,
    generations=200,
    crossover_rate=0.8,
    mutation_rate=0.2,
    tournament_size=3,
    elite_size=2,
    verbose=True
) -> TTPSolution:
    
    clear_path_cache()
    precomputed = PrecomputedData(problem)
    targets = [n for n in problem.graph.nodes if n != 0]
    
    # --- 1. SMART INITIALIZATION ---
    population = []
    
    # A. Radial "Sweep" Sort (Crucial for Depot-centric problems)
    # Sort cities by angle around the depot (0.5, 0.5).
    # This groups angular sectors together, perfect for the Split algorithm.
    depot_pos = problem.graph.nodes[0]['pos']
    def get_angle(node_idx):
        pos = problem.graph.nodes[node_idx]['pos']
        return math.atan2(pos[1] - depot_pos[1], pos[0] - depot_pos[0])
    
    sweep_route = sorted(targets, key=get_angle)
    population.append(TTPSolution(sweep_route, problem.graph))
    
    # B. Nearest Neighbor Heuristic (Greedy distance)
    curr = 0
    nn_route = []
    unvisited = set(targets)
    while unvisited:
        nxt = min(unvisited, key=lambda x: precomputed.get_distance(curr, x))
        nn_route.append(nxt)
        unvisited.remove(nxt)
        curr = nxt
    population.append(TTPSolution(nn_route, problem.graph))

    # C. Random (Fill the rest)
    while len(population) < population_size:
        # Create a shuffled version of the sweep route to maintain some locality
        # but introduce diversity
        route = sweep_route[:]
        
        # Heavy perturbation (swap 30% of cities)
        for _ in range(len(route) // 3):
            i, j = random.sample(range(len(route)), 2)
            route[i], route[j] = route[j], route[i]
            
        population.append(TTPSolution(route, problem.graph))

    # Evaluate Initial Pop
    if verbose: print("Evaluating initial population...")
    for ind in population:
        evaluate_solution_split(ind, problem, precomputed)

    best_ever = max(population, key=lambda ind: ind.fitness)
    if verbose: print(f"Initial best cost: {-best_ever.fitness:.2f}")

    # --- 2. EVOLUTION LOOP ---
    for generation in range(generations):
        
        # Elitism
        elites = elitism_selection(population, elite_size)
        
        # Offspring
        offspring = []
        while len(offspring) < population_size - elite_size:
            parent1 = tournament_selection(population, tournament_size)
            parent2 = tournament_selection(population, tournament_size)
            
            if random.random() < crossover_rate:
                child = order_crossover(parent1, parent2)
            else:
                child = parent1.copy()
            
            if random.random() < mutation_rate:
                if random.random() < 0.6:
                    child = inversion_mutation(child)
                else:
                    child = swap_mutation(child)
            
            offspring.append(child)
        
        # Evaluation
        for ind in offspring:
            evaluate_solution_split(ind, problem, precomputed)
        
        population = elites + offspring
        
        # --- 3. MEMETIC LOCAL SEARCH (The Secret Sauce) ---
        # Every 10 generations, try to strictly improve the best individual
        # using a simple 2-opt hill climber.
        if generation % 10 == 0:
            best_curr = max(population, key=lambda ind: ind.fitness)
            improved_sol = apply_2opt(best_curr, problem, precomputed, max_steps=200)
            
            if improved_sol.fitness > best_ever.fitness:
                best_ever = improved_sol.copy()
                # Inject back into population to spread the good genes
                population[-1] = improved_sol
        
        # Update Global Best
        gen_best = max(population, key=lambda ind: ind.fitness)
        if gen_best.fitness > best_ever.fitness:
            best_ever = gen_best.copy()
            if verbose:
                print(f"Gen {generation}: New best cost = {-best_ever.fitness:.2f}")

    if verbose:
        print(f"\nFinal Best Cost: {-best_ever.fitness:.2f}")
    
    return best_ever

def apply_2opt(solution, problem, precomputed, max_steps=100):
    """
    Simple stochastic 2-opt local search.
    Tries to untangle crossing paths to improve the sequence for the Split algorithm.
    """
    route = solution.route[:]
    best_fitness = solution.fitness
    improved = False
    
    # Try random swaps
    for _ in range(max_steps):
        i, j = sorted(random.sample(range(len(route)), 2))
        if i == 0 or j == len(route)-1: continue 
        
        # 2-Opt Swap (Reverse segment)
        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
        
        # Fast check: Just check fitness
        temp_sol = TTPSolution(new_route, problem.graph)
        fit = evaluate_solution_split(temp_sol, problem, precomputed)
        
        if fit > best_fitness:
            route = new_route
            best_fitness = fit
            improved = True
    
    if improved:
        sol = TTPSolution(route, problem.graph)
        evaluate_solution_split(sol, problem, precomputed)
        return sol
    return solution