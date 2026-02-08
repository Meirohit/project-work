import copy
import networkx as nx

class TTPSolution:
    """
    Represents a TTP solution, individual in population
    """
    def __init__(self, route, graph):
        """
        route: list of city indices (permutation, excluding city 0)
        graph: the problem graph (for calculating paths/costs)
        """
        self.route = route
        self.graph = graph
        self.fitness = None     
        self.cost = None     
        self.path_steps = None  # Final path format [(city, gold), ...]
    
    def copy(self):
        """Create a deep copy of this solution"""
        new_sol = TTPSolution(self.route[:], self.graph)
        new_sol.fitness = self.fitness
        new_sol.cost = self.cost
        new_sol.path_steps = self.path_steps[:] if self.path_steps else None
        return new_sol
    
    def __repr__(self):
        return f"TTPSolution(cost={self.cost:.2f}, route={self.route[:5]}...)"