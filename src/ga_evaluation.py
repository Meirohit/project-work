import networkx as nx
from Problem import Problem
from src.ga_solution import TTPSolution
import math

# Cache for shortest paths
_path_cache = {}

def clear_path_cache():
    """Clear the cache when problem changes"""
    global _path_cache
    _path_cache = {}


def get_shortest_path(graph, u, v):
    """Get cached shortest path between nodes"""
    if (u, v) in _path_cache:
        return _path_cache[(u, v)]
    
    try:
        path = nx.shortest_path(graph, source=u, target=v, weight='dist')
        _path_cache[(u, v)] = path
        _path_cache[(v, u)] = path[::-1]
        return path
    except nx.NetworkXNoPath:
        return []


def evaluate_solution(individual: TTPSolution, problem: Problem):
    """
    Evaluate the fitness of a TTP solution
    """
    if individual.fitness is not None:
        return individual.fitness
    
    ordered_cities = individual.route
    graph = individual.graph
    
    total_cost = 0.0
    current_node = 0
    current_load = 0.0
    full_steps = []
    
    gold_map = nx.get_node_attributes(graph, 'gold')
    
    for next_target in ordered_cities:

        gold_at_target = gold_map[next_target]
        
        # Option A: Direct path
        path_direct = get_shortest_path(graph, current_node, next_target)
        cost_direct = problem.cost(path_direct, current_load)
        
        # Option B: Return to depot
        path_to_home = get_shortest_path(graph, current_node, 0)
        cost_to_home = problem.cost(path_to_home, current_load)
        
        path_from_home = get_shortest_path(graph, 0, next_target)
        cost_from_home = problem.cost(path_from_home, 0)                # Empty load
        
        total_detour_cost = cost_to_home + cost_from_home
        
        # Greedy decision
        if cost_direct <= total_detour_cost:
            # Go direct
            for node in path_direct[1:]:
                g = gold_at_target if node == next_target else 0
                full_steps.append((node, g))
            
            total_cost += cost_direct
            current_load += gold_at_target
            current_node = next_target
        else:
            # Return home first
            if current_node != 0:
                for node in path_to_home[1:]:
                    full_steps.append((node, 0))
                total_cost += cost_to_home
            
            for node in path_from_home[1:]:
                g = gold_at_target if node == next_target else 0
                full_steps.append((node, g))
            
            total_cost += cost_from_home
            current_load = gold_at_target
            current_node = next_target
    
    # Final return to depot
    if current_node != 0:
        path_end = get_shortest_path(graph, current_node, 0)
        cost_end = problem.cost(path_end, current_load)
        total_cost += cost_end
        
        for node in path_end[1:]:
            full_steps.append((node, 0))
    
    # Store results
    individual.cost = total_cost
    individual.fitness = -total_cost  # Negative because we minimize cost
    individual.path_steps = full_steps
    
    return individual.fitness


def evaluate_solution_split(individual, problem, precomputed):
    """
    Evaluates a permutation using the Split algorithm (DP).
    Finds the optimal segmentation of the tour into multiple trips.
    """
    if individual.fitness is not None:
        return individual.fitness
        
    route = individual.route
    n = len(route)
    
    # V[i] = Min cost to service the first i cities in the route
    # P[i] = Predecessor index (to reconstruct the trips)
    V = [float('inf')] * (n + 1)
    P = [0] * (n + 1)
    V[0] = 0
    
    # CONSTRAINT: For Beta > 1, trips are short. 
    # Limit search to 15 cities max per trip to speed up GA (O(N) instead of O(N^2))
    MAX_TRIP_SIZE = 15
    
    for i in range(n):
        # We are at city index i (0 to n-1) in the route.
        # This corresponds to state i in V.
        # We try to form a trip from route[i]...route[j-1]
        
        current_load = 0
        trip_cost = 0
        prev_node = 0 # Start at Depot
        
        # Look ahead up to MAX_TRIP_SIZE cities
        for k in range(1, MAX_TRIP_SIZE + 1):
            j = i + k
            if j > n: 
                break
            
            city = route[j-1]
            
            # 1. Travel from prev_node -> city
            dist_leg = precomputed.get_distance(prev_node, city)
            trip_cost += precomputed.calculate_cost(prev_node, city, current_load)
            
            # 2. Pick up item
            current_load += precomputed.get_gold(city)
            prev_node = city
            
            # 3. Calculate cost to return to depot IMMEDIATELY from here
            # This closes a potential trip segment: i -> j
            return_cost = precomputed.calculate_cost(city, 0, current_load)
            
            total_segment_cost = trip_cost + return_cost
            
            # 4. Update Bellman equation
            if V[i] + total_segment_cost < V[j]:
                V[j] = V[i] + total_segment_cost
                P[j] = i

    # Reconstruct the path steps
    individual.cost = V[n]
    individual.fitness = -V[n]
    
    # Reconstruct the actual movement (for visualization/debugging)
    # We backtrack from n to 0 using P
    trips = []
    curr = n
    while curr > 0:
        prev = P[curr]
        # Trip was from prev to curr (indices in route are prev..curr-1)
        trips.append(route[prev:curr])
        curr = prev
    trips.reverse()
    
    # Build detailed path_steps
    full_steps = []
    for trip in trips:
        curr_node = 0
        for city in trip:
            # Add intermediate steps from shortest path cache
            path = precomputed.get_path(curr_node, city)
            for node in path[1:]:
                g = precomputed.get_gold(node) if node == city else 0
                full_steps.append((node, g))
            curr_node = city
        
        # Return to depot
        path_home = precomputed.get_path(curr_node, 0)
        for node in path_home[1:]:
            full_steps.append((node, 0))
            
    individual.path_steps = full_steps
    return individual.fitness