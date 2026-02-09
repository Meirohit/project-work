import numpy as np

class PheromoneMatrix:
    """
    Pheromone matrix for Ant Colony System (ACS)
    """
    def __init__(self, num_cities, initial_pheromone=0.1):
        self.num_cities = num_cities
        self.tau0 = initial_pheromone
        self.pheromone = np.ones((num_cities, num_cities)) * initial_pheromone
        np.fill_diagonal(self.pheromone, 0)
        
        # Minimum pheromone level (for numerical stability)
        self.tau_min = 0.01
        self.tau_max = 10.0
    
    def get(self, i, j):
        """Get pheromone level between cities i and j"""
        return self.pheromone[i, j]
    
    def local_update(self, i, j, rho=0.1):
        """
        Local pheromone update (ACS variant)
        Applied immediately when ant traverses edge
        
        tau[i,j] = (1-rho)*tau[i,j] + rho*tau0
        """
        new_val = (1 - rho) * self.pheromone[i, j] + rho * self.tau0
        self.pheromone[i, j] = np.clip(new_val, self.tau_min, self.tau_max)
        self.pheromone[j, i] = self.pheromone[i, j]
    
    def global_update(self, best_tour, best_cost, rho=0.1):
        """
        Global pheromone update (only best ant deposits)
        
        tau[i,j] = (1-rho)*tau[i,j] + rho*delta_tau
        where delta_tau = 1/best_cost
        """
        # Evaporate
        self.pheromone *= (1 - rho)
        
        # Deposit on best tour
        delta_tau = 1.0 / best_cost
        
        for i in range(len(best_tour) - 1):
            city_from = best_tour[i]
            city_to = best_tour[i + 1]
            
            self.pheromone[city_from, city_to] += delta_tau
            self.pheromone[city_to, city_from] += delta_tau
        
        # Clip to bounds
        self.pheromone = np.clip(self.pheromone, self.tau_min, self.tau_max)