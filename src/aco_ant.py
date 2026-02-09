import random
import networkx as nx
import numpy as np
from src.aco_solution import ACOSolution

class Ant:
    """
    An ant constructs a solution by visiting cities
    """
    def __init__(self, problem, pheromone_matrix, alpha=1.0, beta=2.0):
        """
        problem: Problem instance
        pheromone_matrix: PheromoneMatrix instance
        alpha: pheromone importance (from your notes)
        beta: heuristic importance (from your notes)
        """
        self.problem = problem
        self.graph = problem.graph
        self.pheromone = pheromone_matrix
        self.alpha = alpha  # Pheromone weight
        self.beta = beta    # Heuristic weight
        
        self.current_city = 0
        self.current_load = 0.0
        self.unvisited = set(range(1, len(self.graph.nodes)))  # All cities except depot
        self.solution = None
        
        # Cache for paths
        self.path_cache = {}
    
    def get_shortest_path(self, u, v):
        """Get cached shortest path"""
        if (u, v) in self.path_cache:
            return self.path_cache[(u, v)]
        
        path = nx.shortest_path(self.graph, source=u, target=v, weight='dist')
        self.path_cache[(u, v)] = path
        return path
    
    def calculate_heuristic(self, city):
        """
        Heuristic: How attractive is this city?
        
        "heuristic[i][j] = gold[j] / (distance[i][j] * (1 + current_weight))"
        
        We want:
        - High gold amount = more attractive
        - Short distance = more attractive
        - Low current load = more attractive (avoid weight penalty)
        """
        gold = self.graph.nodes[city]['gold']
        
        try:
            path = self.get_shortest_path(self.current_city, city)
            distance = nx.path_weight(self.graph, path, weight='dist')
        except:
            return 0.0
        
        if distance == 0:
            return 0.0
        
        attractiveness = gold / (distance * (1 + self.current_load / 1000))
        
        return attractiveness
    
    def select_next_city(self):
        """
        Probabilistic city selection based on pheromone and heuristic
        
        P[i][j] = (pheromone[i][j]^alpha * heuristic[i][j]^beta) / Σ(...)
        """
        if not self.unvisited:
            return None
        
        # Calculate probabilities for each unvisited city
        probabilities = []
        cities = list(self.unvisited)
        
        for city in cities:
            # Pheromone level
            tau = self.pheromone.get(self.current_city, city)
            
            # Heuristic value
            eta = self.calculate_heuristic(city)
            
            # Combined probability (from your notes formula)
            if eta > 0:
                prob = (tau ** self.alpha) * (eta ** self.beta)
            else:
                prob = tau ** self.alpha  # Only pheromone if heuristic is zero
            
            probabilities.append(prob)
        
        # Normalize probabilities
        total = sum(probabilities)
        if total == 0:
            # Fallback: uniform random
            return random.choice(cities)
        
        probabilities = [p / total for p in probabilities]
        
        # Roulette wheel selection
        selected = random.choices(cities, weights=probabilities, k=1)[0]
        return selected
    
    def decide_gold_amount(self, city):
        """
        Decide how much gold to collect at this city
        
        Strategy: decide according to the value of the beta
        """
        available_gold = self.graph.nodes[city]['gold']
        
        # Check if taking all gold would be too expensive
        # if self.problem.beta > 1.5:
        #     return available_gold * 0.5  # Take 50%
        
        # Default: take all gold
        return available_gold
    
    def should_return_to_depot(self, next_city=None):
        """
        Decide if ant should return to depot to unload
        
        Heuristic: Return if weight penalty is becoming too high
        """
        if self.current_load == 0:
            return False
        
        # Estimate cost of continuing with current load
        if next_city:
            try:
                path = self.get_shortest_path(self.current_city, next_city)
                distance = nx.path_weight(self.graph, path, weight='dist')
                marginal_cost = (self.problem.alpha * distance * self.current_load) ** self.problem.beta
                
                # Threshold: higher beta = lower threshold (return more often)
                threshold = 100 / (self.problem.beta ** 1.2)
                
                return marginal_cost > threshold
            except:
                return False
        
        return False
    
    def construct_solution(self):
        """
        Build a complete solution by visiting all cities
        """
        from src.aco_solution import ACOSolution
        
        solution = ACOSolution()
        
        self.current_city = 0
        self.current_load = 0.0
        self.unvisited = set(range(1, len(self.graph.nodes)))
        
        visited_order = [0]  # Track path for pheromone deposit
        
        while self.unvisited:
            # Select next city
            next_city = self.select_next_city()
            
            if next_city is None:
                break
            
            # Check if should return to depot first
            if self.should_return_to_depot(next_city):
                # Return to depot (unload)
                visited_order.append(0)
                solution.add_visit(0, 0, return_to_depot=True)
                self.current_city = 0
                self.current_load = 0.0
            
            # Decide how much gold to collect
            gold_amount = self.decide_gold_amount(next_city)
            
            # Visit the city
            solution.add_visit(next_city, gold_amount)
            visited_order.append(next_city)
            
            # Update state
            self.current_load += gold_amount
            self.current_city = next_city
            self.unvisited.remove(next_city)
        
        # Return to depot at end
        visited_order.append(0)
        
        # Store the path for pheromone update
        solution.visited_order = visited_order
        
        self.solution = solution
        return solution

def evaluate_aco_solution(solution: ACOSolution, problem):
    """
    Calculate the cost of an ACO solution
    Similar to GA evaluation but works with ACO route format
    """
    
    total_cost = 0.0
    current_city = 0
    current_load = 0.0
    path_steps = []
    
    path_cache = {}
    
    def get_path(u, v):
        if (u, v) in path_cache:
            return path_cache[(u, v)]
        path = nx.shortest_path(problem.graph, u, v, weight='dist')
        path_cache[(u, v)] = path
        return path
    
    for city, gold in solution.route:
        if city == 0:
            # Return to depot (unload)
            if current_city != 0:
                path = get_path(current_city, 0)
                cost = problem.cost(path, current_load)
                total_cost += cost
                
                for node in path[1:]:
                    path_steps.append((node, 0))
                
                current_city = 0
                current_load = 0.0
        else:
            # Visit city and collect gold
            path = get_path(current_city, city)
            cost = problem.cost(path, current_load)
            total_cost += cost
            
            for node in path[1:]:
                g = gold if node == city else 0
                path_steps.append((node, g))
            
            current_city = city
            current_load += gold
    
    # Final return to depot
    if current_city != 0:
        path = get_path(current_city, 0)
        cost = problem.cost(path, current_load)
        total_cost += cost
        
        for node in path[1:]:
            path_steps.append((node, 0))
    
    solution.total_cost = total_cost
    solution.path_steps = path_steps
    
    return total_cost
