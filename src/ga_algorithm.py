import random
from Problem import Problem
from src.hybrid_aco.precompute import PrecomputedData
from src.hybrid_aco.beta_optimizer import FastBetaOptimizer
from src.ga_solution import TTPSolution
from src.ga_operators import (
    order_crossover, swap_mutation, 
    inversion_mutation, insert_mutation
)
from src.ga_selection import tournament_selection, elitism_selection
from src.ga_evaluation import evaluate_solution, clear_path_cache


def genetic_algorithm(
    problem: Problem,
    population_size=100,
    generations=200,
    crossover_rate=0.8,
    mutation_rate=0.2,
    tournament_size=3,
    elite_size=2,
    verbose=True
) -> TTPSolution:
    """
    Genetic Algorithm for TTP
    """
    
    clear_path_cache()
    # PRECOMPUTE (this is the key optimization!)
    precomputed = PrecomputedData(problem)
    optimizer = FastBetaOptimizer(precomputed)
    # Step 1: Initialize population (random permutations)
    if verbose:
        print("Initializing population...")
    
    targets = [n for n in problem.graph.nodes if n != 0]
    population = []
    
    for _ in range(population_size):
        route = targets[:]
        random.shuffle(route)
        individual = TTPSolution(route, problem.graph)
        population.append(individual)
    
    # Step 2: Evaluate initial population
    for individual in population:
        #print(f"Individual: {i}")
        evaluate_solution(individual, problem)
    
    # Track best solution
    best_ever = max(population, key=lambda ind: ind.fitness)
    
    if verbose:
        print(f"Initial best cost: {-best_ever.fitness:.2f}")
    
    # Step 3: Evolution loop
    for generation in range(generations):
        
        # Elitism: keep best individuals
        elites = elitism_selection(population, elite_size)
        
        # List for offspring
        offspring = []
        
        while len(offspring) < population_size - elite_size:
            # Parent tournament selection
            parent1 = tournament_selection(population, tournament_size)
            parent2 = tournament_selection(population, tournament_size)
            
            # Crossover
            if random.random() < crossover_rate:
                child = order_crossover(parent1, parent2)
            else:
                child = parent1.copy()
            
            # Mutation
            if random.random() < mutation_rate:
                mutation_type = random.choice([
                    swap_mutation,
                    inversion_mutation,
                    insert_mutation
                ])
                child = mutation_type(child, mutation_rate=1.0)  # Always mutate if chosen
            
            offspring.append(child)
        
        # Evaluate offspring
        for individual in offspring:
            evaluate_solution(individual, problem)
        
        # Replacement: elites + offspring
        population = elites + offspring
        
        # Update best
        generation_best = max(population, key=lambda ind: ind.fitness)
        if generation_best.fitness > best_ever.fitness:
            best_ever = generation_best.copy()
            if verbose:
                print(f"Gen {generation}: New best cost = {-best_ever.fitness:.2f}")
        
        # Monitor the progress
        if verbose and generation % 20 == 0:
            avg_fitness = sum(ind.fitness for ind in population) / len(population)
            print(f"Gen {generation}: Best={-best_ever.fitness:.2f}, Avg={-avg_fitness:.2f}")
    
    if verbose:
        print(f"\nBest cost after all GA: {-best_ever.fitness:.2f}")
    
    # Beta optimization (with early stopping)
    
    return best_ever