import networkx as nx
from Problem import Problem
from src.ga_solution import TTPSolution

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