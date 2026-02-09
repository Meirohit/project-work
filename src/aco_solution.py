class ACOSolution:
    """
    Represents a solution built by an ant
    """
    def __init__(self):
        self.route = []           # Order of cities visited [(city, gold_collected), ...]
        self.returns = []         # Positions where ant returned to depot
        self.total_cost = None
        self.total_gold = 0.0
        self.path_steps = []      # Final format for output
    
    def add_visit(self, city, gold_amount, return_to_depot=False):
        """Add a city visit to the route"""
        self.route.append((city, gold_amount))
        self.total_gold += gold_amount
        
        if return_to_depot:
            self.returns.append(len(self.route) - 1)
    
    def __repr__(self):
        return f"ACOSolution(cost={self.total_cost}, cities={len(self.route)}, returns={len(self.returns)})"