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
class GA_FastBetaOptimizer:
    """
    Fast beta optimization adapted for GA TTPSolution objects.
    Splits a single tour into k-trips to minimize cost when beta > 1.
    """
    def __init__(self, precomputed_data):
        self.data: PrecomputedData = precomputed_data
    
    def optimize(self, solution: TTPSolution, max_k=15) -> TTPSolution:
        """
        Optimizes the given GA solution by trying to split the route into k trips.
        Updates the solution in-place if a better configuration is found.
        """
        route = solution.route
        n = len(route)
        if n == 0:
            return solution

        # Limit search space
        max_k = min(max_k, n, 20)
        
        best_cost = float('inf')
        best_k = 1
        best_trip_grouping = None # Will store List[List[int]] (list of trips)
        
        # 1. Iterate through possible number of trips (k)
        # Early stopping logic: if cost doesn't improve for 3 consecutive k, stop
        no_improvement_count = 0
        current_best_found_here = float('inf')

        for k in range(1, max_k + 1):
            cost, grouping = self._evaluate_k_split(route, k)
            
            if cost < current_best_found_here:
                improvement = current_best_found_here - cost
                current_best_found_here = cost
                
                # Global update
                if cost < best_cost:
                    best_cost = cost
                    best_k = k
                    best_trip_grouping = grouping
                
                no_improvement_count = 0
                
                # Heuristic: if k > 1 and improvement is negligible, stop
                if k > 1 and improvement < 1.0:
                    break
            else:
                no_improvement_count += 1
                if no_improvement_count >= 3:
                    break
        
        # 2. If optimization found a better cost than the original GA evaluation, update solution
        current_solution_cost = -solution.fitness if solution.fitness is not None else float('inf')
        
        if best_cost < current_solution_cost:
            solution.cost = best_cost
            solution.fitness = -best_cost
            # Reconstruct the detailed path steps (node-by-node) for the best grouping
            solution.path_steps = self._reconstruct_detailed_path(best_trip_grouping)
            
        return solution

    def _evaluate_k_split(self, route, k):
        """
        Calculates cost of splitting the route into k groups (trips).
        Returns: (total_cost, list_of_trips)
        """
        n = len(route)
        group_size = n // k
        remainder = n % k
        
        total_cost = 0.0
        trips = []
        start_idx = 0
        
        for i in range(k):
            # Distribute remainder to the first few groups to keep sizes balanced
            trip_size = group_size + (1 if i < remainder else 0)
            end_idx = start_idx + trip_size
            
            trip_cities = route[start_idx:end_idx]
            trips.append(trip_cities)
            
            # Calculate cost for this specific trip (Depot -> Cities -> Depot)
            trip_cost = self._calculate_trip_cost(trip_cities)
            total_cost += trip_cost
            
            start_idx = end_idx
            
        return total_cost, trips

    def _calculate_trip_cost(self, cities):
        """
        Calculates high-level cost for a single trip starting and ending at 0
        """
        current_node = 0
        current_load = 0.0
        trip_cost = 0.0
        
        for city in cities:
            # Add travel cost
            trip_cost += self.data.calculate_cost(current_node, city, current_load)
            
            # Pick up gold
            gold = self.data.get_gold(city)
            current_load += gold
            
            current_node = city
            
        # Return to depot
        trip_cost += self.data.calculate_cost(current_node, 0, current_load)
        
        return trip_cost

    def _reconstruct_detailed_path(self, trip_grouping):
        """
        Reconstructs the full sequence of (node, gold) steps including intermediate nodes
        on shortest paths. Matches the format expected by TTPSolution.path_steps.
        """
        full_steps = []
        
        for trip_cities in trip_grouping:
            current_node = 0
            
            # 1. Go through cities in the trip
            for target_city in trip_cities:
                # Get full path steps from current to target (using precomputed shortest paths)
                path_nodes = self.data.get_path(current_node, target_city)
                
                # Append steps. path_nodes includes [start, ..., end]. 
                # We skip start (index 0) because it was the end of the previous segment.
                for node in path_nodes[1:]:
                    gold_amount = self.data.get_gold(node) if node == target_city else 0
                    full_steps.append((node, gold_amount))
                
                current_node = target_city
            
            # 2. Return to depot after trip is done
            path_home = self.data.get_path(current_node, 0)
            for node in path_home[1:]:
                # No gold collected on return trip
                full_steps.append((node, 0))
                
        return full_steps