def evaluate_tour_fast(tour, gold_collected, precomputed):
    """
    Fast evaluation using precomputed distances and vectorized cost
    """
    total_cost = 0.0
    current = 0
    load = 0.0
    
    cities = [c for c in tour if c != 0]
    
    for city in cities:
        # Use precomputed cost calculation
        cost = precomputed.calculate_cost(current, city, load)
        total_cost += cost
        
        load += gold_collected.get(city, 0)
        current = city
    
    # Return home
    total_cost += precomputed.calculate_cost(current, 0, load)
    
    return total_cost