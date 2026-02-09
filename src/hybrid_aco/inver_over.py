import random
import copy

def inver_over_operator(tour, reference_tour=None, prob_ref=0.85):
    """
    Inver-Over operator for TSP
    
    From your notes (Chapter 7): "Inver over: One gene is selected in the first parent.
    One edge is taken from the second parent, trying to preserve edges of the first."
    
    Args:
        tour: Current tour (list of city indices)
        reference_tour: Reference tour (another solution) or None
        prob_ref: Probability of using reference tour
    
    Returns:
        Improved tour
    """
    if len(tour) < 4:
        return tour
    
    new_tour = tour[1:-1]  # Exclude depot at start/end
    n = len(new_tour)
    
    # Select random city c
    c_idx = random.randint(0, n - 1)
    c = new_tour[c_idx]
    
    # Select c' (next city)
    if reference_tour and random.random() < prob_ref:
        # Use reference tour to find c'
        # Find c in reference tour
        ref_cities = reference_tour[1:-1]  # Exclude depot
        
        if c in ref_cities:
            c_ref_idx = ref_cities.index(c)
            # Get neighbor of c in reference tour
            c_prime = ref_cities[(c_ref_idx + 1) % len(ref_cities)]
        else:
            c_prime = random.choice([city for city in new_tour if city != c])
    else:
        # Random selection
        c_prime = random.choice([city for city in new_tour if city != c])
    
    # Find positions
    if c_prime not in new_tour:
        return tour  # Safety check
    
    c_prime_idx = new_tour.index(c_prime)
    
    # Perform inversion between c and c'
    if c_idx == c_prime_idx:
        return tour
    
    start, end = sorted([c_idx, c_prime_idx])
    
    # Invert segment
    new_tour[start:end+1] = reversed(new_tour[start:end+1])
    
    # Reconstruct full tour with depot
    return [0] + new_tour + [0]


def inver_over_local_search(tour, population, iterations=50):
    """
    Apply Inver-Over multiple times for local optimization
    
    Args:
        tour: Current best tour
        population: List of other tours (for reference)
        iterations: Number of improvement attempts
    
    Returns:
        Improved tour
    """
    current = copy.copy(tour)
    
    for _ in range(iterations):
        # Select reference tour from population
        if population and len(population) > 1:
            reference = random.choice([t for t in population if t != current])
        else:
            reference = None
        
        # Apply operator
        candidate = inver_over_operator(current, reference, prob_ref=0.85)
        
        # Accept (we'll evaluate later)
        current = candidate
    
    return current