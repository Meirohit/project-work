import numpy as np

class PheromoneMatrix:
    """
    Manages pheromone trails between cities
    """
    def __init__(self, num_cities, initial_pheromone=1.0):
        """
        Create pheromone matrix
        
        pheromone[i][j] = pheromone level on edge from city i to city j
        """
        self.num_cities = num_cities
        self.pheromone = np.ones((num_cities, num_cities)) * initial_pheromone    #All start equal (1.0) - no bias initially
        
        # No pheromone on self-loops (no self-loops)
        np.fill_diagonal(self.pheromone, 0)
    
    def get(self, i, j):
        """Get pheromone level between cities i and j"""
        return self.pheromone[i, j]
    
    def evaporate(self, rho=0.1):
        """
        Evaporate pheromones (decay over time)
        
        From your notes: "All pheromones decay by a factor (1 - ρ)"
        rho: evaporation rate (0.1 = 10% evaporation)
        """
        self.pheromone *= (1 - rho)
        
        # Ensure minimum pheromone level (avoid zero)
        self.pheromone = np.maximum(self.pheromone, 0.01)
    
    def deposit(self, route, amount):
        """
        Deposit pheromone along a route
        
        route: list of city indices [0, 3, 7, 2, 5, 0]
        amount: how much pheromone to deposit (usually 1/cost)
        """
        for i in range(len(route) - 1):
            city_from = route[i]
            city_to = route[i + 1]
            self.pheromone[city_from, city_to] += amount
            # Symmetric (undirected graph)
            self.pheromone[city_to, city_from] += amount
    
    def deposit_on_edges(self, edges, amount):
        """
        Deposit pheromone on specific edges
        
        edges: list of (city_i, city_j) tuples
        """
        for city_i, city_j in edges:
            self.pheromone[city_i, city_j] += amount
            self.pheromone[city_j, city_i] += amount