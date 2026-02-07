from Problem import Problem
import networkx as nx
import random
import copy
import random
import networkx as nx
from src.helper_functions import create_neighbor


def solution(p: Problem):
    
    print("Problem solution started")
    # --- 1. Precompute / Caching for Speed ---
    # Calculating shortest paths repeatedly in a loop is slow.
    # We use a dictionary to cache paths between node pairs.
    path_cache = {}

    def get_shortest_path(u, v):
        """Returns the shortest path sequence of nodes between u and v."""
        if (u, v) in path_cache:
            return path_cache[(u, v)]
        
        # Calculate and cache
        try:
            path = nx.shortest_path(p.graph, source=u, target=v, weight='dist')
            path_cache[(u, v)] = path
            path_cache[(v, u)] = path[::-1] # The reverse path is valid too
            return path
        except nx.NetworkXNoPath:
            # Should not happen given the problem statement says graph is connected
            return []

    # --- 2. Cost Calculator & Decoder ---
    def calculate_solution_cost(ordered_cities):
        print("calculate_solution_cost")
        """
        Decodes a permutation of targets into a full valid path and calculates cost.
        Returns: (Total Cost, Formatted Step List)
        """
        if not ordered_cities:
            return 0.0, []

        total_cost = 0.0
        current_node = 0
        current_load = 0.0
        
        # Result list format: [(node_visited, gold_collected), ...]
        full_steps = []
        
        gold_map = nx.get_node_attributes(p.graph, 'gold')

        for next_target in ordered_cities:
            gold_at_target = gold_map[next_target]
            
            # --- Option A: Direct Path (Current -> Next) ---
            path_direct = get_shortest_path(current_node, next_target)
            cost_direct = p.cost(path_direct, current_load)
            
            # --- Option B: Return to Depot (Current -> 0 -> Next) ---
            path_to_home = get_shortest_path(current_node, 0)
            cost_to_home = p.cost(path_to_home, current_load)
            
            path_from_home = get_shortest_path(0, next_target)
            cost_from_home = p.cost(path_from_home, 0) # Empty load
            
            total_detour_cost = cost_to_home + cost_from_home

            # --- Greedy Decision ---
            if cost_direct <= total_detour_cost:
                # GO DIRECT
                # Append intermediate steps (skipping the first node as we are already there)
                for node in path_direct[1:]:
                    # If this node is the target, we collect gold, otherwise 0
                    g = gold_at_target if node == next_target else 0
                    full_steps.append((node, g))
                
                total_cost += cost_direct
                current_load += gold_at_target
                current_node = next_target
                
            else:
                # RESET AT HOME
                # 1. Go Home
                if current_node != 0:
                    for node in path_to_home[1:]:
                        full_steps.append((node, 0)) # Drop gold at 0 is implicit by logic, explicit by visiting
                    total_cost += cost_to_home
                
                # 2. Go to Target
                for node in path_from_home[1:]:
                    g = gold_at_target if node == next_target else 0
                    full_steps.append((node, g))
                    
                total_cost += cost_from_home
                current_load = gold_at_target # We only have the new gold
                current_node = next_target

        # --- Final Return to Depot ---
        if current_node != 0:
            path_end = get_shortest_path(current_node, 0)
            cost_end = p.cost(path_end, current_load)
            total_cost += cost_end
            
            for node in path_end[1:]:
                full_steps.append((node, 0))
                
        return total_cost, full_steps

    # --- 3. Hill Climbing Initialization ---
    targets = list(p.graph.nodes)
    if 0 in targets: targets.remove(0)
    
    # Initial Solution: Random Shuffle
    current_permutation = targets[:]
    random.shuffle(current_permutation)
    
    best_permutation = current_permutation[:]
    best_cost, best_steps = calculate_solution_cost(best_permutation)
    print()
    print("Initial")
    print(f"initial_permutation: {best_permutation}")
    print(f"initial_cost: {best_cost}")
    print(f"initial_steps: {best_steps}")
    print()
    
    # --- 4. Optimization Loop ---
    max_iterations = 2500
    
    for _ in range(max_iterations):
        # Create Neighbor: Swap 2 cities
        neighbor_permutation = best_permutation[:]
        i, j = random.sample(range(len(neighbor_permutation)), 2)
        neighbor_permutation[i], neighbor_permutation[j] = neighbor_permutation[j], neighbor_permutation[i]
        
        # Evaluate
        neighbor_cost, neighbor_steps = calculate_solution_cost(neighbor_permutation)
        
        # Accept if better
        if neighbor_cost < best_cost:
            best_cost = neighbor_cost
            best_permutation = neighbor_permutation
            best_steps = neighbor_steps

    print(f"Final Best Cost: {best_cost}")
    return best_steps



if __name__ == "__main__":
    p = Problem(100, density=0.2, alpha=1, beta=1)
    print(f"\nBaseline cost: {p.baseline():.2f}")
    hc_steps = solution(p=p)
    print(F"\nGreedy Hill Climbing path: {hc_steps}")
    