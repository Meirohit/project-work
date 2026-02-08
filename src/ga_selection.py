import random

def tournament_selection(population, tournament_size=3):
    """
    Tournament Selection
    "Pick N members at random then select the best among them"
    """
    tournament = random.sample(population, tournament_size)
    winner = max(tournament, key=lambda ind: ind.fitness)
    return winner

def elitism_selection(population, elite_size=2):
    """
    Select the best N individuals (elitism)
    They are copied unmodified as offspring"
    """
    sorted_pop = sorted(population, key=lambda ind: ind.fitness, reverse=True)
    return sorted_pop[:elite_size]