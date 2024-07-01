# Algorithms

## Table of Contents
1. [Greedy Algorithm](#Greedy-Algorithm)


## Greedy Algorithm
<details>
  <summary>Scheduling, Minimum Spanning Tree(Prim's, Kruskal's)</summary>

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
      - Find-Union `O(E logV)`
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
        
  **Find Union**
  - Kruskal's algorithm union until we have 1 clustering
  - Max-Spacing k-clusterings
    - Separated pairs are those who are assigned to different clusterings
    - Goal: minimize the spacing between nearest separated paris
    - Solution: apply union-find until we have k clusterings
  - Different implementations
    - Lazy union: directly update parent
    - Union by rank to avoid deep tree
      - rank = height of the tree
        - rank(node) = max(rank(children of node)) + 1
      - higher rank will be parent (no need to update rank)
      - increment parent's rank if two tree has the same height
      ```python
      def union(self, p, q):
          rootP = self.find(p)
          rootQ = self.find(q)
  
          if rootP != rootQ:
              if self.rank[rootP] > self.rank[rootQ]:
                  self.parent[rootQ] = rootP
              elif self.rank[rootP] < self.rank[rootQ]:
                  self.parent[rootP] = rootQ
              else:
                  self.parent[rootQ] = rootP
                  self.rank[rootP] += 1
      ```
    - Path compression
      - update parent to root parent after calling `find(node)`
      ```python
      def find(self, p):
          if self.parent[p] != p:
              self.parent[p] = self.find(self.parent[p])  # Path compression
          return self.parent[p]
      ```
      - rank of parent will change after many path compressions but we don't update it
  - Time complexity
    - Find = O(n) for lazy union | O(log n) for rank union
    - Union = O(Find + 1) for both
    - Total = O(n log n)
  - More advanced topics: Hopcroft-Ullman analysis, Ackermann function, Tarjan's analysis, 
</details>

<details>
  <summary>Huffman Codes (Prefix-Free Codes), Dynamic Programing, Weighted Independent Sets</summary>

  **Huffman Code**
  - Prefix-free codes: no codeword is a prefix of any other codeword
    - Optimal for Variable-Length Coding: Prefix-free codes are often used in Huffman coding, which is an optimal variable-length coding scheme.
    - Ex. {0, 10, 110, 111} for A, B, C, D and if p(A) = .6, p(B) = .25, p(C) = .1, p(D) = .005. Then avg bits = .6 * 1 + .25 * 2 + .1 * 3 + .005 * 3 = 1.415
    - **Codes as Tree**
      - no label in internal nodes. Only at leaf nodes => no char is an ancestor of the other => prefix-free
      - \# bits of encoding = height of the tree = `ceiling(log<sub>2</sub>(# characters))` for balanced tree
    - Application: Variable-length encoding (ex. MP3 encoding)
  - Goal = minimize avg bits given sets of character frequencies (not alwasy balanced tree). `p_i` = probability of i's character and `d_i` = depth of i's character in the tree
    $$L(T) = \sum_{i=0}^n p_i d_i $$ 
  - **Greedy**
    - merge two nodes with lowest frequency to form a new node whose frequency = sum of two children's frequency. 
    ```python
    def buildTree(leafs):
        heapify(leafs) # lowest frequency first
        while(leafs.hasTwo):
            first = leafs.pop()
            second = leafs.pop()
            new_node = merge(first, second) // make two the children of new_node.
            new_node.freq = first.freq + second.freq
            leafs.push(new_node)
    ```

  **Dynamic Programming**
  - Weighed Independent Sets (WIS): House Robber DP
    - IS = no adjacent vertices in the set
    - goal = maximize the weights
    ```python
    def weighted_independent_set(vertices, weights):
        n = len(vertices)
        dp = [0] * (n + 1)
        dp[1] = weights[0]
        
        for i in range(2, n + 1):
            dp[i] = max(dp[i-1], dp[i-2] + weights[i-1])

        // construct independent set backtracking the dp's value
        return dp[n]
    ```
</details>

- 
