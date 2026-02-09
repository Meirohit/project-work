import random
from Problem import Problem
from src.hybrid_aco.precompute import PrecomputedData
from src.hybrid_aco.pheromone import PheromoneMatrix
from src.hybrid_aco.ant import FastPackingAnt
from src.hybrid_aco.inver_over import inver_over_operator
from src.hybrid_aco.beta_optimizer import FastBetaOptimizer
from src.hybrid_aco.fast_evaluation import evaluate_tour_fast

def fast_hybrid_aco_ttp(
    problem: Problem,
    num_ants=20,           
    num_iterations=50,     
    alpha=1.0,
    beta=2.5,
    q0=0.9,
    rho_global=0.1,
    inver_over_prob=0.5,   
    optimize_trips=True,
    verbose=True
):
    """
    Optimized hybrid ACO for speed
    
    Key optimizations:
    1. Precompute all paths and distances
    2. Vectorized heuristic calculations
    3. Reduced iterations
    4. Selective Inver-Over application
    5. Early stopping in beta optimization
    """
    
    if verbose:
        print("=" * 70)
        print("FAST HYBRID ACO FOR TTP")
        print("=" * 70)
    
    # PRECOMPUTE (this is the key optimization!)
    precomputed = PrecomputedData(problem)
    
    num_cities = precomputed.num_cities
    
    # Initialize pheromone
    pheromone = PheromoneMatrix(num_cities, initial_pheromone=0.1)
    
    # Initialize optimizer
    beta_opt = FastBetaOptimizer(precomputed)
    
    # Best solution tracking
    best_tour = None
    best_gold = None
    best_cost = float('inf')
    
    # Population for Inver-Over 
    population = []
    max_population = 20  
    
    if verbose:
        print(f"Running {num_iterations} iterations with {num_ants} ants...")
        print()
    
    # Main loop
    for iteration in range(num_iterations):
        iteration_tours = []
        iteration_gold = []
        iteration_costs = []
        
        # Ants construct solutions
        for ant_id in range(num_ants):
            ant = FastPackingAnt(precomputed, pheromone, alpha, beta, q0)
            tour, gold = ant.construct_solution_fast()
            
            # Fast evaluation
            cost = evaluate_tour_fast(tour, gold, precomputed)
            
            iteration_tours.append(tour)
            iteration_gold.append(gold)
            iteration_costs.append(cost)
        
        # Best in iteration
        iter_best_idx = iteration_costs.index(min(iteration_costs))
        iter_best_tour = iteration_tours[iter_best_idx]
        iter_best_gold = iteration_gold[iter_best_idx]
        iter_best_cost = iteration_costs[iter_best_idx]
        
        # Apply Inver-Over selectively (not every iteration)
        if random.random() < inver_over_prob and len(population) > 0:
            # Only a few iterations
            for _ in range(15): 
                reference = random.choice(population)
                iter_best_tour = inver_over_operator(
                    iter_best_tour, 
                    reference, 
                    prob_ref=0.85
                )
            
            # Re-evaluate
            refined_cost = evaluate_tour_fast(iter_best_tour, iter_best_gold, precomputed)
            if refined_cost < iter_best_cost:
                iter_best_cost = refined_cost
        
        # Update population (keep small and diverse)
        population.append(iter_best_tour)
        if len(population) > max_population:
            population.pop(0)
        
        # Global pheromone update
        pheromone.global_update(iter_best_tour, iter_best_cost, rho=rho_global)
        
        # Update global best
        if iter_best_cost < best_cost:
            best_tour = iter_best_tour
            best_gold = iter_best_gold
            best_cost = iter_best_cost
            
            if verbose:
                print(f"Iter {iteration}: New best = {best_cost:.2f}")
        
        # Progress (less frequent)
        if verbose and iteration % 10 == 0 and iteration > 0:
            print(f"Iter {iteration}: Best={best_cost:.2f}")
    
    # Beta optimization (with early stopping)
    if optimize_trips and problem.beta > 1.0:
        if verbose:
            print("\nBeta optimization...")
        
        opt_cost, opt_k, opt_plan = beta_opt.optimize_trips_fast(
            best_tour,
            best_gold,
            max_k=15  # Limited search
        )
        
        if verbose:
            print(f"Optimized: {opt_k} trips, cost={opt_cost:.2f}")
        
        if opt_cost < best_cost:
            best_cost = opt_cost
            best_plan = opt_plan
    else:
        # Construct simple plan
        best_plan = construct_simple_plan_fast(best_tour, best_gold, precomputed)
    
    # Convert to output
    path_steps = plan_to_path_format(best_plan)
    
    if verbose:
        print(f"\nFINAL COST: {best_cost:.2f}")
        print("=" * 70)
    
    return path_steps, best_cost


def construct_simple_plan_fast(tour, gold_collected, precomputed):
    """Fast simple plan construction"""
    plan = []
    cities = [c for c in tour if c != 0]
    
    for city in cities:
        gold = gold_collected.get(city, 0)
        plan.append([(city, gold)])
    
    plan.append([(0, 0)])
    return plan


def plan_to_path_format(trip_plan):
    """Convert to output format"""
    path_steps = []
    for trip in trip_plan:
        for city, gold in trip:
            path_steps.append((city, gold))
    
    if not path_steps or path_steps[-1] != (0, 0):
        path_steps.append((0, 0))
    
    return path_steps