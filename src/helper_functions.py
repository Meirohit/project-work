import random
from Problem import Problem
import networkx as nx


def analyze_problem(p: Problem):
    """Analyze the problem instance"""
    graph = p.graph
    
    print(f"Number of cities: {len(graph.nodes)}")
    print(f"Number of edges: {len(graph.edges)}")
    print(f"Alpha: {p.alpha}, Beta: {p.beta}")
    
    total_gold = sum(graph.nodes[n]['gold'] for n in range(1, len(graph.nodes)))
    print(f"Total gold to collect: {total_gold:.2f}")
    
    # Analyze distances from city 0
    distances = nx.single_source_dijkstra_path_length(graph, 0, weight='dist')
    print(f"Average distance from city 0: {sum(distances.values())/len(distances):.4f}")
    print(f"Max distance from city 0: {max(distances.values()):.4f}")
    
    return total_gold, distances


def calculate_total_cost(p:Problem, solution_path):
    total_cost = 0.0
    current_load = 0.0
    current_node = 0  # Start at (0,0)
    
    # The solution_path is a list of tuples: [(node, gold_collected), ...]
    for next_node, collected_gold in solution_path:
        
        # Calculate cost to move from current_node to next_node
        # Note: We pass [current_node, next_node] as the path list to p.cost
        segment_cost = p.cost([current_node, next_node], current_load)
        total_cost += segment_cost
        
        # Update state
        current_node = next_node
        
        if current_node == 0:
            current_load = 0  # Reset load if we are at the depot
        else:
            current_load += collected_gold # Add gold picked up
            
    return total_cost


def inspect_graph(p:Problem):
    print(f"{'From':<5} {'To':<5} {'Distance':<10} {'Gold at Destination':<20}")
    print("-" * 50)
    
    # Iterate over all edges in the graph
    # data=True allows us to access the edge attributes (like 'dist')
    for u, v, data in p.graph.edges(data=True):
        dist = data.get('dist', 0.0)
        
        # Get gold at both nodes (since graph is undirected, you can travel both ways)
        gold_u = p.graph.nodes[u]['gold']
        gold_v = p.graph.nodes[v]['gold']
        
        # Print Edge U -> V
        print(f"{u:<5} {v:<5} {dist:<10.4f} {gold_v:<20.4f}")
        
        # If you want to see the reverse perspective (V -> U) explicitly:
        # print(f"{v:<5} {u:<5} {dist:<10.4f} {gold_u:<20.4f}")


def create_neighbor(permutation, operator='random'):
    """Multiple neighborhood operators"""
    neighbor = permutation[:]
    
    if operator == 'swap' or operator == 'random' and random.random() < 0.4:
        # Your current method
        i, j = random.sample(range(len(neighbor)), 2)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
    
    elif operator == '2opt' or operator == 'random' and random.random() < 0.4:
        # Reverse a segment (very effective for TSP)
        i, j = sorted(random.sample(range(len(neighbor)), 2))
        neighbor[i:j+1] = reversed(neighbor[i:j+1])
    
    elif operator == 'insert':
        # Remove city and insert elsewhere
        i = random.randint(0, len(neighbor) - 1)
        city = neighbor.pop(i)
        j = random.randint(0, len(neighbor))
        neighbor.insert(j, city)
    
    return neighbor

