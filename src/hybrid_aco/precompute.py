import networkx as nx
import numpy as np
from Problem import Problem

class PrecomputedData:
    """
    Precompute all expensive calculations once
    """
    def __init__(self, problem: Problem):
        self.problem = problem
        self.graph = problem.graph
        self.num_cities = len(problem.graph.nodes)
        self.alpha = problem.alpha
        self.beta = problem.beta
        self.gold = nx.get_node_attributes(self.graph, 'gold')
        
        print("Precomputing shortest paths and distances...")
        
        # Precompute ALL shortest paths
        self.all_paths = {}
        self.all_distances = np.zeros((self.num_cities, self.num_cities))
        
        for i in range(self.num_cities):
            paths = nx.single_source_dijkstra_path(self.graph, i, weight='dist')
            lengths = nx.single_source_dijkstra_path_length(self.graph, i, weight='dist')
            
            for j in range(self.num_cities):
                if i != j:
                    self.all_paths[(i, j)] = paths[j]
                    self.all_distances[i, j] = lengths[j]
        
        # Precompute gold amounts
        # self.gold = {i: self.graph.nodes[i]['gold'] for i in range(self.num_cities)}
        
        print("Precomputation complete!")
    
    def get_path(self, i, j):
        """Get precomputed path"""
        return self.all_paths.get((i, j), [i, j])
    
    def get_distance(self, i, j):
        """Get precomputed distance"""
        return self.all_distances[i, j]
    
    def get_gold(self, city):
        """Get gold at city"""
        return self.gold.get(city, 0)
    
    def calculate_cost(self, i, j, load):
        """
        Fast cost calculation using precomputed distance
        
        cost = distance + (alpha * distance * load)^beta
        """
        dist = self.all_distances[i, j]
        return dist + (self.problem.alpha * dist * load) ** self.problem.beta