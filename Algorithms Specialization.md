# Algorithms

## Table of Contents
1. [Greedy Algorithm](#Greedy-Algorithm)


## Greedy Algorithm
**Motivation**
- Internet routing: shortest path (negative cost -> Dijkstra's algorithm doesn't work)
- Sequence alignement: how similar two sequence are = minimize total penalty = P<sub>alternate</sub> + P<sub>missing</sub>
- Optimal caching
- Compared to Divide & Conquer
  - easy to implement
  - better runing time
  - hard to prove correctness: via iterative induction, "exchange argument" (prove by contradiction, exchange optimal to our greedy)
 
**Scheduling**
- Definition: many jobs to schedule. Each job j has weight w<sub>j</sub>, length l<sub>j</sub>, completion time c<sub>j</sub> = sum of job lengths up to and scheduling j.
- Objective Funciton = Goal
  - minimize the weighted sum of completion times (w<sub>j</sub>*c<sub>j</sub>)
- Greedy
  - Preferred: smaller length + larger weight
  - If w<sub>i</sub> > w<sub>j</sub>, and l<sub>i</sub> > l<sub>j</sub>, then choose larger ratio = w<sub>i</sub> / l<sub>i</sub>

**Minimum Spanning Tree**
- Definition: connect all vertices together with minimal cost (doesn't have to be a path)
- Applications to clustering:
    - Max-spacing k-clustering
    - 
- **Prim's**: min-heap => `O(E + V logV) = O(E logV)` to insert vertices to priority queue
  - randomly pick a V, expand one unconnected V by choosing the nearest/cheapest adjacent one
  - cut property: `For any cut (S, V-S) of the graph, if there exists an edge e = (u, v) such that u is in set S and v is in set V-S, then e is a safe edge for the MST.`
  - Pseudocode 
    ```
    Input: Graph G with vertices V and edges E, starting vertex s
    
    MSTSet = {s} // Start with the starting vertex in the MST set
    key[] = {INFINITY, INFINITY, ..., INFINITY} // Initialize key values to INFINITY
    parent[] = {-1, -1, ..., -1} // Array to store the parent of each vertex in the MST
    
    key[s] = 0 // Set key value of starting vertex to 0
    
    pq = priority_queue // to store vertices not yet included in MST, sorted by key values.
    while(pq):
        u = pq.pop() // with the minimum key value from the priority queue.
        MSTSet.add(u)
        foreach(v: adj[u]): //for all adjacent vertices of u, that are not in MSTSet, we update the min_distance, and add it to pq (to check its vertices later)
            if v is not in MSTSet and weight[u][v] < key[v]:
                update key[v] = weight[u][v]
                parent[v] = u
                pq.add({weight[u][v], v})

    Output the MST using parent[] which stores the MST edges.
    ```
- **Kruskal's**: 
    - union-find `O(E logV)`
    - Pseudocode
      ```
      KruskalMST(graph):
          MST = {}
          pq = priority_queue
          foreach(edge e: E): pq.push(e)
      
          for each edge (u, v) pq:
              if Find(u) ≠ Find(v): // If u and v are in different sets (no cycle is formed)
                  MST.add(edge (u, v))
                  Union(u, v) // Combine sets of u and v
        
          return MST
      ```
- 
