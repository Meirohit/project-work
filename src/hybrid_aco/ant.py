import random
import numpy as np
import networkx as nx
from Problem import Problem


class FastPackingAnt:
    """
    Optimized ant using precomputed data
    """
    def __init__(self, precomputed, pheromone_matrix, alpha=1.0, beta=2.5, q0=0.9):
        self.data = precomputed
        self.pheromone = pheromone_matrix
        self.alpha = alpha
        self.beta = beta
        self.q0 = q0
    
    def calculate_heuristics_vectorized(self, current, unvisited, current_load):
        """
        Vectorized heuristic calculation for all unvisited cities at once
        """
        cities = np.array(list(unvisited))
        
        # Get distances for all unvisited cities at once
        distances = self.data.all_distances[current, cities]
        
        # Get gold amounts
        golds = np.array([self.data.get_gold(c) for c in cities])
        
        # Weight factor
        weight_factor = 1.0 + (current_load / (self.data.num_cities + 1e-6))
        
        # Vectorized heuristic: (gold + 1) / (distance * weight_factor)
        eta = (golds + 1.0) / (distances * weight_factor + 1e-6)
        
        return cities, eta
    
    def select_next_city_fast(self, current, unvisited, current_load):
        """
        Fast city selection using vectorized operations
        """
        cities, eta = self.calculate_heuristics_vectorized(current, unvisited, current_load)
        
        # Get pheromones for all candidates
        tau = np.array([self.pheromone.get(current, c) for c in cities])
        
        # Calculate values: tau^alpha * eta^beta
        values = (tau ** self.alpha) * (eta ** self.beta)
        
        if random.random() < self.q0:
            # EXPLOITATION: Choose best
            best_idx = np.argmax(values)
            return cities[best_idx]
        else:
            # EXPLORATION: Probabilistic
            total = values.sum()
            if total < 1e-10:
                return random.choice(cities)
            
            probabilities = values / total
            selected_idx = np.random.choice(len(cities), p=probabilities)
            return cities[selected_idx]
    
    def dynamic_packing_fast(self, city, current_load, num_remaining):
        """
        Simplified fast packing decision
        """
        available_gold = self.data.get_gold(city)
        return available_gold
    
    def construct_solution_fast(self):
        """
        Fast solution construction
        """
        current = 0
        unvisited = set(range(1, self.data.num_cities))
        tour = [0]
        gold_collected = {}
        current_load = 0.0
        
        while unvisited:
            # Fast selection
            next_city = self.select_next_city_fast(current, unvisited, current_load)
            
            if next_city is None:
                break
            
            # Local pheromone update
            self.pheromone.local_update(current, next_city, rho=0.1)
            
            # Fast packing
            gold_amount = self.dynamic_packing_fast(
                next_city,
                current_load,
                len(unvisited) - 1
            )
            
            tour.append(next_city)
            gold_collected[next_city] = gold_amount
            current_load += gold_amount
            
            current = next_city
            unvisited.remove(next_city)
        
        tour.append(0)
        
        return tour, gold_collected