# Algorithms

## Table of Contents
1. [Divide and Conquer, Sorting and Searching, and Randomized Algorithms](#)
2. [Graph Search, Shortest Paths, and Data Structures](#)
3. [Greedy & DP](#Greedy-Algorithms-&-Dynamic-Programming): Scheduling, MST, Huffman; WIS, Knapsack, Sequence Alignment, Optimal BST
4. [Shortest Paths Revisited, NP-Complete Problems and What To Do About Them](#)


## Divide and Conquer, Sorting and Searching, and Randomized Algorithms
<details>
  <summary></summary>

  
</details>

## Graph Search, Shortest Paths, and Data Structures
<details>
  <summary></summary>

  
</details>

## Greedy Algorithms & Dynamic Programming
<details>
  
  <summary>Greedy: Scheduling, Minimum Spanning Tree(Prim's, Kruskal's), Huffman Codes</summary>
  
  \
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
        
  - **Find Union**
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
</details>

<details>
  <summary>Dynamic Programing: Weighted Independent Sets, Knapsack, Sequence Alignment, Optimal Binary Search Tree</summary>

  \
  **Weighed Independent Sets (WIS): House Robber DP**
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
  **Knapsack**
  - each item has weight $w_i$ and values $v_i$
  - goal = maximize benefit while total weight is under capacity C
  ```python
  def knapsack(weights, values, capacity):
      n = len(values)
      # Create a 2D array to store the maximum value that can be attained with the given capacity
      dp = [[0 for x in range(capacity + 1)] for x in range(n + 1)]
  
      # Build the dp array from bottom up
      for i in range(n + 1):
          for w in range(capacity + 1):
              if i == 0 or w == 0:
                  dp[i][w] = 0
              elif weights[i - 1] <= w:
                  dp[i][w] = max(values[i - 1] + dp[i - 1][w - weights[i - 1]], dp[i - 1][w])
              else:
                  dp[i][w] = dp[i - 1][w]
  
      # The maximum value that can be attained with the given capacity
      return dp[n][capacity]
  ```
  **Sequence Alignment (Edit Distance)**
  - two strings X (length m) and Y (length n), missmatch of each character = 1 penalty
  - goal = match two strings and minimize penalty
  ```python
  def needleman_wunsch(seq1, seq2, match_score=1, mismatch_penalty=-1, gap_penalty=-1):
      # Create a scoring matrix
      n = len(seq1)
      m = len(seq2)
      score_matrix = [[0] * (m + 1) for _ in range(n + 1)]
  
      # Initialize the scoring matrix
      for i in range(1, n + 1):
          score_matrix[i][0] = gap_penalty * i
      for j in range(1, m + 1):
          score_matrix[0][j] = gap_penalty * j
  
      # Fill the scoring matrix
      for i in range(1, n + 1):
          for j in range(1, m + 1):
              match = score_matrix[i - 1][j - 1] + (match_score if seq1[i - 1] == seq2[j - 1] else mismatch_penalty)
              delete = score_matrix[i - 1][j] + gap_penalty
              insert = score_matrix[i][j - 1] + gap_penalty
              score_matrix[i][j] = max(match, delete, insert)
      # Perform the traceback to get the optimal alignment
      return traceback(seq1, seq2, score_matrix, match_score, mismatch_penalty, gap_penalty)

  def traceback(seq1, seq2, score_matrix, match_score, mismatch_penalty, gap_penalty):
      align1, align2 = '', ''
      i, j = len(seq1), len(seq2)
      
      while i > 0 and j > 0:
          current_score = score_matrix[i][j]
          if current_score == score_matrix[i - 1][j - 1] + (match_score if seq1[i - 1] == seq2[j - 1] else mismatch_penalty):
              align1 = seq1[i - 1] + align1
              align2 = seq2[j - 1] + align2
              i -= 1
              j -= 1
          elif current_score == score_matrix[i - 1][j] + gap_penalty:
              align1 = seq1[i - 1] + align1
              align2 = '-' + align2
              i -= 1
          else:
              align1 = '-' + align1
              align2 = seq2[j - 1] + align2
              j -= 1
  
      while i > 0:
          align1 = seq1[i - 1] + align1
          align2 = '-' + align2
          i -= 1
  
      while j > 0:
          align1 = '-' + align1
          align2 = seq2[j - 1] + align2
          j -= 1
  
      return align1, align2
  ```
  **Optimal Binary Search Tree**
  - given probability of all search keys (assume only successful searches) 
  - find the best search tree whilch minimizes the weighted/avg search time (search nodes including itself) $$C(T) = \sum_{i=0}^n p_i search_i $$
  - if all $p_i$ are the same, we can use balanced tree
  - difference between this and huffman codes
    - internal nodes can be used as search key in optimal binary search tree
    - search key value must obey the basic rule of binary search tree (bigger value on the right node)
  ```
  cost = [[0 for x in range(n+1)] for y in range(n+1)]
  for i in range(n):
      cost[i][i] = freq[i]
  # optCost_memoized(freq, 0, n - 1)

  def optCost_memoized(freq, i, j):
      if cost[i][j]:
          return cost[i][j]
   
      # Get sum of freq[i], freq[i+1], ... freq[j]
      fsum = Sum(freq, i, j)
   
      # Initialize minimum value
      Min = 999999999999
   
      for r in range(i, j + 1):
          c = (optCost_memoized(freq, i, r - 1) + optCost_memoized(freq, r + 1, j))
          Min = min(c, Min)
   
      # Return minimum value
      return cost[i][j] = fsum + Min
  ```
</details>

## Shortest Paths Revisited, NP-Complete Problems and What To Do About Them
<details>
  <summary></summary>

  
</details>
