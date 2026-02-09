import networkx as nx
import numpy as np
from src.ga_solution import TTPSolution
from src.hybrid_aco.precompute import PrecomputedData
    
class FastBetaOptimizer:
    """
    Fast beta optimization with early stopping
    """
    def __init__(self, precomputed: PrecomputedData):
        self.data = precomputed
    
    def evaluate_k_trips_fast(self, tour, gold_collected, k):
        """
        Fast k-trip evaluation
        """
        cities = [c for c in tour if c != 0]
        n = len(cities)
        
        if k < 1 or k > n:
            return float('inf'), []
        
        # Divide cities into k groups
        group_size = n // k
        remainder = n % k
        
        total_cost = 0.0
        trip_plan = []
        start_idx = 0
        
        for i in range(k):
            trip_size = group_size + (1 if i < remainder else 0)
            end_idx = start_idx + trip_size
            
            trip_cities = cities[start_idx:end_idx]
            
            # Evaluate this trip
            trip_cost = 0.0
            current = 0
            load = 0.0
            path_seq = []
            
            for city in trip_cities:
                # Fast cost calculation
                trip_cost += self.data.calculate_cost(current, city, load)
                gold = gold_collected.get(city, 0)
                load += gold
                path_seq.append((city, gold))
                current = city
            
            # Return to depot
            trip_cost += self.data.calculate_cost(current, 0, load)
            path_seq.append((0, 0))
            
            trip_plan.append(path_seq)
            total_cost += trip_cost
            start_idx = end_idx
        
        return total_cost, trip_plan
    
    def optimize_trips_fast(self, tour, gold_collected, max_k=15):
        """
        Fast trip optimization with early stopping
        """
        cities = [c for c in tour if c != 0]
        n = len(cities)
        
        if n == 0:
            return 0.0, 1, [[(0, 0)]]
        
        max_k = min(max_k, n, 15)  # Limit search space
        
        best_cost = float('inf')
        best_k = 1
        best_plan = []
        
        # Early stopping: if cost doesn't improve for 3 consecutive k, stop
        no_improvement_count = 0
        
        for k in range(1, max_k + 1):
            cost, plan = self.evaluate_k_trips_fast(tour, gold_collected, k)
            
            if cost < best_cost:
                improvement = best_cost - cost
                best_cost = cost
                best_k = k
                best_plan = plan
                no_improvement_count = 0
                
                # Early stop if improvement is tiny
                if k > 1 and improvement < 1.0:
                    break
            else:
                no_improvement_count += 1
                
                if no_improvement_count >= 3:
                    break  # Stop early
        
        return best_cost, best_k, best_plan
    
    ### Below for GA format ###
    def evaluate_partition(self, tour, k):
        """
        Splits the tour into k roughly equal trips.
        Returns (total_cost, full_path_sequence)
        """
        cities = tour # tour is already without 0
        n = len(cities)
        if k < 1: k = 1
        if k > n: k = n
        
        # Split cities into k chunks
        # e.g., 100 cities, k=3 -> 34, 33, 33
        chunk_size = n // k
        remainder = n % k
        
        total_cost = 0.0
        full_path = []
        
        start_idx = 0
        for i in range(k):
            # Determine size of this trip
            size = chunk_size + (1 if i < remainder else 0)
            end_idx = start_idx + size
            trip_nodes = cities[start_idx:end_idx]
            
            # --- Simulate Trip ---
            # 1. Depot -> First City
            current_node = 0
            current_load = 0.0
            
            # Path from Depot to First
            first_city = trip_nodes[0]
            # Add steps (excluding start node 0 to avoid duplicates in plotting)
            path_to_first = self.data.get_path(0, first_city)
            for node in path_to_first[1:]:
                g = self.data.get_gold(node) if node == first_city else 0
                full_path.append((node, g))
                
            total_cost += self.data.calc_step_cost(0, first_city, 0)
            current_load += self.data.get_gold(first_city)
            current_node = first_city
            
            # 2. Inter-city travel
            for next_city in trip_nodes[1:]:
                path_seg = self.data.get_path(current_node, next_city)
                # Add steps
                for node in path_seg[1:]:
                    g = self.data.get_gold(node) if node == next_city else 0
                    full_path.append((node, g))
                
                total_cost += self.data.calc_step_cost(current_node, next_city, current_load)
                current_load += self.data.get_gold(next_city)
                current_node = next_city
            
            # 3. Return to Depot
            path_home = self.data.get_path(current_node, 0)
            for node in path_home[1:]:
                full_path.append((node, 0))
                
            total_cost += self.data.calc_step_cost(current_node, 0, current_load)
            
            start_idx = end_idx
            
        return total_cost, full_path

    def optimize(self, individual: TTPSolution):
        """
        Finds the best number of trips (k) for the given route.
        Updates the individual in-place.
        """
        n = len(individual.route)
        
        # Heuristic: search range for k (number of trips)
        # If beta is high, we expect more trips.
        # We search specific points to save time, or a range.
        
        best_cost = float('inf')
        best_k = 1
        best_path = []
        
        # Search strategy: Check 1..10, then every 5 up to N/2
        # (You can adjust this search space for speed vs accuracy)
        search_space = list(range(1, 15)) + list(range(15, n, 5))
        
        # Early stopping variables
        prev_cost = float('inf')
        worse_count = 0
        
        for k in search_space:
            if k > n: break
            
            cost, path = self.evaluate_partition(individual.route, k)
            
            if cost < best_cost:
                best_cost = cost
                best_k = k
                best_path = path
                worse_count = 0 # Reset counter
            else:
                worse_count += 1
            
            # If we see 3 consecutive K values getting worse, stop searching
            if worse_count >= 3:
                break
                
        individual.cost = best_cost
        individual.fitness = -best_cost
        individual.path_steps = best_path
        return individual.fitness