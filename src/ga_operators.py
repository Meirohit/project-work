import random
import copy
from src.ga_solution import TTPSolution

def order_crossover(parent1: TTPSolution, parent2: TTPSolution) -> TTPSolution:
    """
    Order Crossover (OX) - preserves relative order
    """
    
    route1 = parent1.route
    route2 = parent2.route
    size = len(route1)
    
    # Select two random crossover points
    cx_point1, cx_point2 = sorted(random.sample(range(size), 2))
    
    # Copy segment from parent1
    child_route = [None] * size
    child_route[cx_point1:cx_point2] = route1[cx_point1:cx_point2]
    
    # Fill remaining positions with cities from parent2 (in order)
    pointer = cx_point2
    for city in route2[cx_point2:] + route2[:cx_point2]:
        if city not in child_route:
            if pointer >= size:
                pointer = 0
            child_route[pointer] = city
            pointer += 1
    
    return TTPSolution(child_route, parent1.graph)

def swap_mutation(individual: TTPSolution, mutation_rate: float = 0.1) -> TTPSolution:
    """
    Swap two random cities
    """
    
    if random.random() > mutation_rate:
        return individual
    
    mutant = individual.copy()
    
    if len(mutant.route) < 2:
        return mutant
    
    i, j = random.sample(range(len(mutant.route)), 2)
    mutant.route[i], mutant.route[j] = mutant.route[j], mutant.route[i]
    
    # Invalidate fitness
    mutant.fitness = None
    mutant.cost = None
    
    return mutant


def inversion_mutation(individual: TTPSolution, mutation_rate: float = 0.1) -> TTPSolution:
    """
    2-opt mutation: Reverse a segment of the route, Minimize changes in edges (most neighbours stay neighbours)
    """
    if random.random() > mutation_rate:
        return individual
    
    mutant = individual.copy()
    
    if len(mutant.route) < 2:
        return mutant
    
    i, j = sorted(random.sample(range(len(mutant.route)), 2))
    mutant.route[i:j+1] = reversed(mutant.route[i:j+1])
    
    mutant.fitness = None
    mutant.cost = None
    
    return mutant


def insert_mutation(individual: TTPSolution, mutation_rate: float = 0.1) -> TTPSolution:
    """
    Remove a city and insert it elsewhere, Minimize changes in the order, preserve relative order"
    """
    
    if random.random() > mutation_rate:
        return individual
    
    mutant = individual.copy()
    
    if len(mutant.route) < 2:
        return mutant
    
    i = random.randint(0, len(mutant.route) - 1)
    city = mutant.route.pop(i)
    j = random.randint(0, len(mutant.route))
    mutant.route.insert(j, city)
    
    mutant.fitness = None
    mutant.cost = None
    
    return mutant