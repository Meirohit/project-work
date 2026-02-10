# Traveling Thief Problem (TTP) Solution

This project implements and compares three metaheuristic approaches to solve the **Traveling Thief Problem (TTP)**. The TTP combines the Traveling Salesman Problem (TSP) and the Knapsack Problem (KP), requiring the optimization of a route visiting all cities while managing the weight of collected items, which exponentially increases travel costs.



## Problem Description

The goal is to visit $N$ cities, collect all gold, and return to the start (depot). The challenge lies in the interdependence of the subproblems: the weight of collected gold slows down the thief, making the sequence of retrieval critical.

**Cost Function:**
$$Cost = distance + (\alpha \times distance \times weight)^\beta$$

* **$\beta > 1$**: High weight penalty. The cost grows exponentially with weight, requiring strategies like frequent returns to the depot to unload.
* **$\beta \le 1$**: Low weight penalty. The problem behaves more similarly to a standard TSP.

## Implemented Algorithms

1.  **Genetic Algorithm (GA) [Default/Enhanced]**
    * **Logic:** An evolutionary approach that optimizes the permutation of cities.
    * **Enhancement:** Uses a **Split Algorithm (Dynamic Programming)** to mathematically determine the optimal locations to return to the depot. Includes **Geometric Initialization** (Sweep/MST) to handle high-complexity instances.
    * **Performance:** Consistently outperforms other methods, especially on high $\beta$ instances.

2.  **Hybrid Ant Colony Optimization (Hybrid ACO)**
    * **Logic:** A constructive approach using pheromones and heuristics, enhanced with local search.
    * **Key Features:** Includes **Precomputation** for speed, **Inver-Over Local Search**, and **Beta Optimization** (post-processing trip splitting).

3.  **Basic Ant Colony Optimization (ACO)**
    * **Logic:** Standard ACO implementation.
    * **Status:** Serves as a baseline; generally struggles with high $\beta$ values due to exponential cost explosion during construction.

## Project Structure

* `s336521.py`: **Main entry point.** Contains the `solution()` function and algorithm selector.
* `Problem.py`: Defines the TTP problem instance, graph generation, and cost function.
* `src/`: Directory containing useful code for algorithm implementations.

## How to run

1. Ensure you have Python 3.x installed along with the required dependencies. 
2. Configure the Algorithm: Open `s336521.py`. Navigate to Line 20 inside the solution() function and set the algorithm variable to your desired method::

```python
algorithm = "GA"     # Options: "ACO", "GA", "HYBRID"
```
3. Finally, simply run the script via the terminal:
```bash
python s336521.py
```
You can modify the problem instance parameters in the `__main__` block of `s336521.py`:

```python
p = Problem(num_cities=100, density=1, alpha=1, beta=3)
```
    