# Algorithms

This project focuses on exploring and implementing various fundamental algorithmic problems and challenges. Each problem presents unique computational tasks, often requiring efficient solutions to handle large datasets or complex scenarios. Below is a summary of the key problems addressed in this project:


1. [Longest Common Subsequence](#Longest-Common-Subsequence)
2. [Subset Sum](#Subset-Sum)
3. [Knapsack Problem](#Knapsack-Problem)
4. [Homework Scores](#Homework-Scores)
5. [Maximize Happiness](#Maximize-Happiness)
6. [Edit Distance](#Edit-Distance)
7. [Shortest Paths](#Shortest-Paths)
8. [Complexity Problems](#Complexity)
9. [Hamiltonian Path](#Hamiltonian-Path)
10. [Traveling Salesman Problem](#TSP)
11. [Vertex Cover](#Vertex-Cover)
12. [Other Easy vs Hard Problems](#Easy-vs-Hard-Problems)
13. [Interval Problem](#Interval-Problem)
14. [Counting Inversions](#Counting-inversions)
15. [Selection Problem](#Selection-Problem)
16. [Integer Multiplication](#Integer-Multiplication)
17. [Minima-Set Problem](#Minima-Set-Problem)

## Longest Common Subsequence
- Usage: DNA

Recursion
```python
def LCS(n, m):
    if 0 in [n, m]:
        return 0
    elif X[n] == Y[m]:
        return 1 + LCS(n-1, m-1)
    else:
        return max(LCS(n-1, m), LCS(n, m-1))
```
- no need for LCS(n-1, m-1) because next level it will be called twice 
- memorization ⇒ 2d array since n and m are both changing

DP
```python
def LCS(n, m):
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if X[i] == Y[j]:
                dp[i][j] = 1 + dp[i-1][j-1]
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
```

## Subset Sum
Recursion
```python
def SubsetSum(n, target):
    if target == 0:
        return True
    if n == 0:
        return False
    return SubsetSum(n-1, target) or (target - S[n] >= 0 and SubsetSum(n-1, target - S[n]))
```
DP
```python
def SubsetSum(n, target):
    sub = [[False] * (target + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        sub[i][0] = True
    for i in range(1, target + 1):
        sub[0][i] = False
    for i in range(1, n + 1):
        for j in range(1, target + 1):
            sub[i][j] = sub[i-1][j] or (target - S[i] >= 0 and sub[i-1][target - S[i]])
    return sub[n][target]
```

## Knapsack Problem
Find the maximal benefit value `V` such that a subset of items `{1…i}` can be taken with the weight at most `W`.
```python
# R → maximum benefit value that comes from the first n items with maximum weight W
def R(n, W):
    if n == 0 or W == 0:
        return 0
    if w[n] <= W:
        return max(R(n-1, W), b[n] + R(n-1, W - w[n]))
    else:
        return R(n-1, W)

def DP(n, W):
    ben = [[0] * (W + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, W + 1):
            if w[i] <= W:
                ben[i][j] = max(ben[i-1][j], ben[i-1][j - w[i]] + b[i])
            else:
                ben[i][j] = ben[i-1][j]
    return ben[n][W]
```

Find the smallest knapsack weight `W` such that a subset of items `{1…i}` can be taken with value at least `V`.
```python
def R(i, V):  # returns smallest weight at i item
    if i <= 0 and V > 0:
        return float('inf')  # reach the end and V is not satisfied
    
    # skip i item or take it
    if prefixSum[i-1] < V:
        return R(i-1, max(V - v[i], 0)) + W[i]
    else:
        return min(R(i-1, V), R(i-1, max(V - v[i], 0)) + W[i])

# Instead of directly min(R(), R()), we can use prefix sum to know if the ith value is mandatory to take

def DP():
    # fill in dp[i][0] with 0
    # fill in dp[0][V != 0] with inf
    for i in range(1, n + 1):
        for j in range(1, V + 1):
            if prefixSum[i-1] < v:
                dp[i][j] = dp[i-1][max(..)] + w[i]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i-1][max(j-v[i], 0)] + w[i])
    
    return min(dp)
```

## Homework Scores
```python
# R → maximum points to get from the first i homeworks
def R(i):
    if i <= 0:
        return 0
    return max(e[i] + R(i-1), h[i] + R(i-2))

def DP(n):
    score = [0] * (n + 1)
    score[1] = h[1] if allowed else e[1]
    for i in range(2, n + 1):
        score[i] = max(e[i] + score[i-1], h[i] + score[i-2])
    return score[n]
```

## Maximize Happiness
```python
# R → maximum happiness value to get from the person and its sub
def R(node):
    if node.children is None:
        return v[node]
    return max(sum(R(c) for c in node.children), v[node] + sum(R(gc) for gc in node.grandchildren))

def R(person, canBeInvited):
    if canBeInvited:
        notInvited = (0, [])
        invited = (v[person], [person])
        for c in person.children:
            notInvited += R(c, canBeInvited)
            invited += R(c, not canBeInvited)
        if notInvited > invited:
            return notInvited
        else:
            return invited
    else:
        notInvited = (0, [])
        for c in person.children:
            notInvited += R(c, not canBeInvited)
        return notInvited

def DP(n):
    hap = [0] * (n + 1)
    for l in leafNodes.num:
        hap[l] = v[l]
    for i in range(n - 1, -1, -1):
        if i in leafNode:
            continue
        hap[i] = max(sum(hap[c] for c in node.children), v[node] + sum(hap[gc] for gc in node.grandchildren))
    return hap[0]
```

## Edit Distance
Given two strings X & Y (not necessarily of equal length), we want to convert the first string to the other by a sequence of insertions, deletions, and substitutions. The cost is the number of operations we perform.
```python
def R(i, j):
    if i == 0:
        return j
    if j == 0:
        return i
    if X[i] == Y[j]:
        return R(i-1, j-1)
    else:
        return 1 + min(R(i-1, j-1), R(i-1, j), R(i, j-1))

# Time complexity: O(n*m) = O(len(X) * len(Y))
def DP():
    for i in range(len(X)):
        for j in range(len(Y)):
            if X[i] == Y[j]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1])
    return dp[-1][-1]

```

## Shortest Paths
given directed graph with weights w_e on edges, shortest path v to T. (NO negative circle)

- What is the longest, in terms of the number of edges, that a shortest path could be? 
- What would it mean if a shorter path had more edges than that? 
- What are the shortest paths ending at T for the following graph?

```python
# Returns minimal cost of the path to vertex v to t (limited to i edges)
def SDSP(i, v):
    if i == 0:
        return 0 if v == t else inf
    else:
        cost = SDSP(i-1, v)
        for w in adj[v]:
            cost = min(cost, SDSP(i-1, w) + cost[w][v])
        return cost

# Time complexity: O(n*m) = O(|V| * |E|)
def DP():
    # Fill in base cases
    
    for i in range(total_number_edges):
        for w in adj[v]:
            dp[i][v] = min(dp[i-1][v], dp[i-1][w] + cost[w][v])
    # Better approach
    if newcost < dp[i-1][v]:
```
- Sum of `O(δ(v)) = m (for a directed graph) = |E|` for filling in all the `dp[i][v]`.

## Complexity
### 3-Sat
Given a set `X` of Boolean variables = `{x1, . . . , xn}`; We have k clauses, each of 3 terms, disjuncted `((x_i1 or x_j1 or x_n1)` and … and `((x_ik or x_jk or x_nk))`. Find the value of X set that k clauses are all true 
### Independence Set
Given a graph `G` and an integer `k`, determine if there is some subset of the vertices `V′`, `|V′| ≥ k` such that no two vertices in `V′` share an edge

```python
if V'   !⊆ V then
    return false
if |V'| ≠ k then
    return false
for all edges e = (u, v) in E:
    if u in V’ and v in V’ then
        return false
return true
```
Independent Set with `k=# of Clauses`
```python
def 3-SAT(n, k):
    for each clause A, B, C:
        create 3 vertices A, B, C
        add edges (A, B), (B, C), (A, C)
    for each variable x_i:
        connect x_i node to all (not x_i) nodes
    return INDEPENDENT SET(G, k)
```
- false positive. Show G has Independent set of size k ⇒ 3-SAT satisfied
- false negative. Show 3-SAT satisfied ⇒ G has independent set with size k

### 3-Color
Given an undirected graph G, determine whether there is a mapping of vertices in G to 3 distinct colors such that no edge is monochromatic
reduce to 3-Sat
- fully connected graph with 3 vertices to represent the color (True, False, neutral)
- fully connected 3-vertices graph connected to neutral (x1, not x1, neutral)...

### Set Cover problem
We are given a series of n sets and a value k, can you select no more than k sets such that every element in the n sets is in at least one element of the k chosen?
- Select the one with maximum length. (2nd row)

| Set number | Elements |
|------------|---------|
| 1	         | A B     |
| 2	         | A C D E |
| 3	         | B C F I |
| 4          | D G     |
| 5          | E	H     |
| 6          | 	I J    |
| 7          | 	F G H J |
-  For every other sets M, make M = M - (2nd row)
    
| Set number | Elements |
| -------- | ------- |
|1	| A|
|3	|B F I|
|4|	G|
|5|	H|
|6|	I J|
|7|	F G H J|

Set Cover - Vertex Cover
- Make a adjacent list for the vertex cover, and perform the exactly the same steps in set cover 
- For each vertex, define a set. 
- For each edge, define a distinct element and add it only to the sets that were created from its endpoints. 
- The resulting set of sets has a set cover of size k iff the graph had a vertex cover of size k.

## Hamiltonian Path
A simple path/cycle that includes every vertex in directed/undirected graph
Use Hamiltonian Path Solution to solve 3-SAT (Reduce 3-SAT to Hamiltonian Path)
For all the possible bool variables <x1, x2…xi>, have i rows just like the following.
![260_1.png](images%2F260_1.png)
- L → R: True 
- R → L: False 
- For every clause, create a clause node outside the row_graph. 
- For every bool variable in this clause, Insert another node in the middle, connect it to the clause node. For x1, left node → clause node → right node. For not x1, right node → clause node → left node.

![260_3.png](images%2F260_3.png)

## TSP
Hamiltonian Cycle of Lowest Total Weight
```python
# TSPJourney(i, A) = cost of the shortest path from vi to v1 passing through each vertex in A exactly once. (vi and v1 are not in A)
# Time complexity: O(n^2 * 2^n)
# O(n) for i range
# O(2^n) for A range
# O(n) to compute (i, A)

def TSPJourney(i, A):
    if len(A) == 1:
        return cost[A[0]][1]
    curMin = inf
    for a in A:
        curMin = min(curMin, cost[i][a] + TSPJourney(a, A - {a}))
    return curMin

totalMin = inf
for i in range(1, A.len):
    totalMin = min(totalMin, cost[A[0]][A[i]] + TSPJourney(i, A))
```

## Vertex Cover
Given a graph G and integer k, determine if there is some subset of the vertices V ′ , |V ′ | ≤ k such that each edge is incident to some vertex in V ′

with k = 3 ⇒ 3 2 7 (Select vertices that are not in vertex cover ⇒ complement of the vertex cover = independent set)

Example: Given S students, B busses, F projects. For i project it takes s_i and b_i resource to get g_i. Maximize sum(g_i).
```python
def R(i, S, B):
    if i == 0:
        return 0
    if s[i] > S or b[i] > B:
        return R(i-1, S, B)
    else:
        return max(R(i-1, S, B), R(i-1, S-s[i], B-b[i]) + g[i])

# Time complexity: O(FBS)

# DP table declaration
declare Goodwill[0..n, 0..B, 0..S]    // to maintain our optimal values
declare DoProject[0..n, 0..B, 0..S]  // to reconstruct whether we are doing the project or not

# Fill base cases
for j = 0 to B:
    for k = 0 to S:
        Goodwill[0, j, k] = 0
        DoProject[0, j, k] = false

# General cases
for i = 1 to F:
    for j = 0 to B:
        for k = 0 to S:
            # Fill in Goodwill[i, j, k] with our defined recurrence
            if b[i] > j or s[i] > k:  // Cannot undertake this project
                Goodwill[i, j, k] = Goodwill[i-1, j, k]
            else:
                Goodwill[i, j, k] = max(Goodwill[i-1, j, k], g[i] + Goodwill[i-1, j - b[i], k - s[i]])
            
            # Check if we undertake the project
            if Goodwill[i, j, k] != Goodwill[i-1, j, k]:
                DoProject[i, j, k] = true
            else:
                DoProject[i, j, k] = false
```

## Easy vs Hard Problems
| Easy Problem                                            | 	Hard Problem                                     |
|---------------------------------------------------------|---------------------------------------------------| 
| no answer + easy to prove                               | no answer + hard to prove                         |
| 2-color (find a cycle with odd vertices)                | 3-color O(3^n)                                    |
| minimum spanning tree                                   | TSP                                               |
| Euler tour (visit all edges exactly once) ⇐ edge degree | Hamiltonian Tour: visit all vertices exactly once |
| shortest path                                           | longest path without re-visiting any vertex       |
| bipartite matching: pair matching without duplicate     | 3D-matching: 3-group matching without duplicate   |
| linear programming	                                     | integer programming                               |
### Graph Coloring

Given a (simple, undirected) graph \( G = (V, E) \):
- Assign each vertex a color.
- Cannot assign the same color to the two endpoints of an edge.
- **Chromatic number** \( \chi(G) \): Minimum distinct colors needed.

### Spanning Tree Problem

Given a connected graph:
- **Minimum Spanning Tree (MST)**: Includes edges with the lowest total weight.
  - Prim's/Dijkstra’s: Start with a visited set and select adjacent nodes with minimal cost.
  - Kruskal’s: Order edges by cost and connect them if their endpoints are in different sets.

 Cycle Property
- **C is any cycle**
- **e is heaviest edge**
- **Any tree T that includes e, not MST**

Cut Property
- **Cut the vertices into X and G – X**
- **e is the lightest edge with endpoints in X and G – X respectively**
- **Any tree that avoids e, not MST**

### Euler and Hamiltonian

- **Hamiltonian Tour (NP-complete)**: Visit every vertex exactly once.
- **Euler Tour**: Cross every edge exactly once.
  - Impossible if the number of vertices with odd degree is more than 2.

### Shortest/Longest Path Problems

- **Shortest Path**
  - Unweighted: BFS
  - Weighted, non-negative cost: Dijkstra’s
  - Weighted, arbitrary cost: Bellman-Ford

### Matching Problem

- **Bipartite Matching**
  - Given undirected bipartite graph \( G = (V, E) \), where \( V = V1 \cup V2 \).
  - **Matching (M)**: Each node appears at most once in \( M \). Find the largest matching.

### Linear and Integer Problem

Example: Web server company wants to buy new servers

- **Standard Model**
  - Cost: $400
  - Power: 300W
  - Two shelves of rack
  - Handles 1000 hits/min

- **Cutting-edge Model**
  - Cost: $1600
  - Power: 500W
  - One shelf
  - Handles 2000 hits/min

- **Budget**
  - $36,800
  - 44 shelves of space
  - 12,200W power

- **Goal**: Maximize the number of hits served per minute (integer solution ⇒ NP Complete)

### Dominating Set

**Definition:**
A subset \( V' \subseteq V \) of the vertices such that every vertex is either in \( V' \) or adjacent to one that is. Prove that determining if a dominating set of size \( k \) is present in a general graph is NP-complete.

- **The earliest legal step may not be in the optimal solution**

```python
def MaxEvent(i, prev):  # return max number of event at i position when previous event is at prev
    if i == n:
        return 1
    if i - prev >= abs(d[i] - d[prev]) and n - i >= abs(d[n] - d[i]):
        return max(1 + MaxEvent(i + 1, i), MaxEvent(i + 1, prev))
    else:
        return MaxEvent(i + 1, prev)

dp = [0] * n
dp[-1] = 1
for i in range(n - 2, -1, -1):
    for j in range(j):
        if abs(d[i] - d[j]) <= abs(j - i) and abs(d[n] - d[i]) <= (n - j):
            dp[i][j] = max(1 + dp[i + 1][i], dp[i + 1][j])
        else:
            dp[i][j] = dp[i + 1][j]
return dp[0][0]
```

### Dominating Set on a Undirected Tree

Each vertex has a cost. Select \( V' \subseteq V \) such that each vertex \( v \in V' \) or there exists \( u \) such that \( (u,v) \in E \) and \( u \in V' \).

- **DOM_y(u):** Min cost dominating set rooted at \( u \) that includes \( u \)
  - \( \text{cost}(u) + \left\{ \sum_{\text{children}} \min(\text{DOM_y}(x), \text{DOM_n}(x)) \right\} \)

- **DOM_n(u):** Min cost dominating set rooted at \( u \) that excludes \( u \)
  - \( \min_{\text{children } c} \left\{ \text{DOM_y(children)} + \sum_{\text{children}} \min(\text{DOM_y}(x), \text{DOM_n}(x)) \right\} \)

### Subset Sum to 3 SAT

- For every variable \( x_i \), \( v_i = 1 \) if \( x_i \), \( v_i' = 1 \) if \( \neg x_i \).
- For every clause \( C_i \) column, put \( S_i = 1 \) and \( S_i' = 2 \).
- For every clause \( C_i \) column, only put \( 1 \) to the \( v \) ( \( v \leftarrow x \) in \( C_i \) ).
- The target value in the end = \( 111\ldots4444\ldots \).

### Choose Reduction

**Constraint Satisfaction → 3 SAT**

- **Definition:** Categorize problem into an existing bucket.

**Clique Problem (CP) ← 3 SAT**

Given an undirected graph \( G \) and an integer \( k \), is there a complete subgraph of \( G \) with \( k \) nodes?

- \( k \) clauses in SAT. Each \( x \) in SAT → vertex in CP.
- Connect vertex in different clause if they are not negation (ex. \( x \) and \( \neg x \)).
- Solution(CP) ⇒ Solution(k-clause 3-SAT).

**Maximum Cut Problem (MCP) ← 3 SAT (TBC)**

Given a graph, the problem is to partition its vertices into two disjoint sets such that the number of edges between the two sets is maximized.

- Each variable \( x_i \) and \( \neg x_i \) in SAT → vertex in MCP and add edge between.
- \( x_i \) is in same clause with \( x_j \) → edge between.

### Other reductions
**Packing → Independent Set**

- **Definition:** Choose at least \( k \) objects (some pair cannot be chosen at the same time).

**Covering → Vertex/Set Cover, Dominating Set**

- **Definition:** Choose at most \( k \) objects to achieve certain goals.

**Permutation/Sequencing → Directed Hamiltonian Path/Cycle**

- **Definition:** Determine an order for \( n \) objects, or if one exists, subject to constraints.
- Scheduling Problem (SP) ← Subset Sum (SS)
  - Each integer \( x \) in SS → a job with processing time = \( x \).
  - Target integer \( t \) in SS → deadline of the job.
  - Solution to SP ⇒ able to generate a solution for SS.
  - SP with order ← directed Hamiltonian path/cycle \( G \).
  - Each vertex \( v \) within \( G \) → job.
  - Directed edge from \( i \) to \( j \) → job \( i \) must be completed before \( j \) starts.
  - If no constraint between two jobs, double arrowed between vertices.
  - Solution\( G \) ⇒ solution\( SP \) to schedule all jobs in the correct order.

- **Longest Common Subsequence Problem (LCS):** Given two sequences of characters, the problem is to find the longest subsequence that is common to both sequences.

**Numerical → Subset Sum**

- **Definition:** Select object(s) subject to a totality constraint.

- **Knapsack Problem (KP) ← Subset Sum (SS)**

  - Given a set of items, each with a weight and a value, and a knapsack of a certain capacity, the problem is to determine the items to include in the knapsack such that the total weight does not exceed the capacity, and the total value is maximized.

  - SS target sum T → capacity in KP.
  - SS integer \( s_i \) in set → item with \( v_i \) and \( w_i \) both equal to \( s_i \).
  - Solution(KP) to fill pack with capacity ⇒ solution(SS) to sum items to T.

- **Bin Packing Problem (BPP) ← Subset Sum (SS)**

  - Given a set of items, each with a size, and a set of bins with fixed capacity, the problem is to pack the items into the bins such that the total number of bins used is minimized.

  - Same as KP ← SS process.
  - Solution(BPP) means we find a way to pack all items to a single box with capacity = T(SS) ⇒ solution(SS) to sum items to T.

**Partitioning → 3 Color**
- **Definition:** Divide a collection into subsets (one object appears in exactly one bucket).

**Sharing Problem ← Subset Sum (SS)**:
- Assume we have SS with input set = S and sum(S) = M, target = T
  - create a new element x_n+1 s.t. solution(SS) + x_n+1 = sum(rest of the set) or solution(SS) = sum(rest of the set) + x_n+1 (depending on which on is larger sum)
  - Assume solution(rest) = M - T is the larger sum. then x_n+1 = M - 2T. 
- SP input = S + x_n+1, and goal of SP is to find two sets that sum up to (M-T)
- If there is a solution for SP, then it’s also the solution for SS
- Algorithm Design
  - Naive Solution Approach: For each connected subgraph \( A \):
    - Call Algorithm A with parameter \( k = \text{size of subgraph} \)
    - Decrease \( k \) until a "yes" answer is found
  - BFS for "Yes"
    - Utilize BFS to explore subgraphs and determine if a solution exists

**Inner Circle (IC) - Vertex Cover (VC)**
- vertices in VC → professors in IC
- edge E between i and j in VC → issue E agreed by professor i and j 
- solution(IC) means we have at least one professor, who correctly voted on each issue, in the solution ⇒ solution(IC) has at least one endpoint for each edge ⇒ this solution(IC) can be used for VC

**Strongly Independent Set (SIS) and NP**
- add a vertex to the middle of every edge. SIS(new graph) = ID(previous graph)
- connect every newly added vertex to every other newly added vertex (so that they are not selected)

**Set Packing ← Independent set (ID)**
- Connect two sets if they share common elements
- ID in a graph represents set packing

**Stingy-SAT and Vertex Cover (VC)**
- Connect variables to sets and determine minimum covering vertices of size \( k \)

**Clustering ←3 Color**
- adjacent vertex in 3C ⇒ distance > T in C
- solution(C) with k = 3, then we have a 3C solution

**Tonian Path (TP) ← directed Hamiltonian path (HP)**
- edges and vertex in HP → edges and vertex in TP 
- solution(TP) with size = n ⇒ solution for HP

**Min-Cost Fast Path (MCFP) ← Hamiltonian path (HP)**
- edge, vertex in HP → edge, vertex in MCFP
- T = n - 1, C = n-1 and t(e) = c(e) = 1 for all edges
- solution(MCFP) = solution(HP) 

## Interval Problem
Unweighted interval scheduling (same credit + different length of courses) and we want max # of courses without overlapping.
- sort courses by end time
- choose that ends earliest

Each homework can be started at s<sub>i</sub>, and estimated work time is t<sub>i</sub>, the deadline is d<sub>i</sub>. The lateness of the homework = s<sub>i</sub> + t<sub>i</sub> - d<sub>i</sub>. Penalty = max(all lateness). 
- sort b

## Counting inversions
The problem is to count the number of inverted pairs `(i, j)` such that `i < j` and `A[i] > A[j]` in an array `A`.
```python
# Running time T(n) = 2 * T(n/2) + n = 4 * T(n/4) + 2*n/2 = …
count = 0
i = 1
j = n/2 + 1
T[1..n]
k = 1

while i <= n/2 and j <= n:
    if A[i] > A[j]:
        j += 1
        count += 1
    T[k] = A[j]
    k += 1
    else:
        i += 1
    T[k] = A[i]
    k += 1

while i <= n/2:
    T[k] = A[i]
    i += 1
    k += 1

while j <= n:
    T[k] = A[j]
    j += 1
    k += 1

copy T to A
return count
```
For divide & conquer algorithms with a recurrence of the form T(n) = a * T(n/b) + f(n), where a ≥ 1, b ≥ 1, and f(n) is asymptotically positive:
1. If there is a small constant ε > 0 such that f(n) is O(n^(log_b a - ε)), then T(n) is θ(n^(log_b a)).
2. If there is a constant k ≥ 0 such that f(n) is θ(n^(log_b a) log^k n), then T(n) is θ(n^(log_b a) log^(k+1) n).
3. If there is a small constant ε > 0 such that f(n) is Ω(n^(log_b a + ε)), then T(n) is θ(f(n)).

Examples:
- T(n) = 4T(n/2) + n → when ε = 0.1 in 1, f(n) = n is in O(n^(2-0.1)) → T(n) is in θ(n^2).
- T(n) = 2T(n/2) + n*logn → when k = 1 in 2, f(n) is in n*logn → T(n) is in θ(n log n).
- T(n) = T(n/3) + n → when ε = 0.1 in 3, f(n) = Ω(n^(0+ε)) → T(n) is in θ(n).
- T(n) = 9T(n/3) + n^2.5 → when ε = 0.1 in 3, f(n) is Ω(n^(2+ε)).


## Selection Problem

**Problem**: Given a list S and a number k, find the k-th smallest number in S.

### Randomized Selection

```python
def quickSelect(S, k):
    if n is small:  # brute force and return
        # implement brute force logic here
        return
    x = random element in S  # pivot
    L = elements smaller than x
    G = elements larger than x
    if k <= |L|:
        quickSelect(L, k)
    elif k == |L| + 1:
        return x
    else:
        quickSelect(G, k - |L| - 1)
```
- worst-running time = O(n^2) with bad pivot (every pivot is next largest/smallest number)
- average = O(n) if pivot is the median ??

Median-of-five (better quickselect?)
- instead of randomly picking x
- divide S into g groups (g = n/5)
- find the median of each group (most groups contain 5 elements) 	← O(1)
- x = median of those medians						← T(n/g)
- n/5 groups has n/5 medians 
  - → n/10 medians smaller than x 
  - → 2n/10 smaller than x and in median groups (not median)
  - → 3n/10 smaller elements in those groups (non-median + median)
  - → 3n/10 larger elements 
  - 4n/10 element unknown


## Integer Multiplication

Given two n-bit integers X and Y, compute X × Y. The algorithm you learned for this in grade school takes time O(n^2).

### Al-Khwarizmi’s Algorithm

X × Y = (X_H × 2^(n/2) + X_L) × (Y_H × 2^(n/2) + Y_L)  
= X_H Y_H 2^n + ((X_H Y_L + X_L Y_H) * 2^(n/2) + X_L Y_L)

```python
def Mult(X, Y):
    Create Xh, Xl, Yh, Yl
    A = Mult(Xh, Yh)
    B = Mult(Xh, Yl)
    C = Mult(Xl, Yh)
    D = Mult(Xl, Yl)
    
    return A * 2^n + (B + C) * 2^(n/2) + D
```
- without E: T(n)=4 T(n/2)+θ(n)=θ(n^2)
- with E: T(n)

## Minima-Set Problem

Given a set S of n points in the plane, we want to find the set of minima points. That is, if we include (x, y) in our output, we want to ensure that there is no point (x', y') in the output such that x ≥ x' and y ≥ y'.

### Divide & Conquer Approach

The divide and conquer algorithm for solving the Minima-Set Problem is outlined as follows:

```python
def MinimaSet(S):
    if n <= 1:
        return S
    
    p = median point in S by x
    L = points less than p
    G = points greater than or equal to p
    
    M1 = MinimaSet(L)
    M2 = MinimaSet(G)
    
    # Select points from M2 whose y is smaller than min(M1.y)
    min_y_M1 = min(M1.y)
    result = M1 + {e in M2 | e.y < min_y_M1}
    
    return result
```
- T(n) = 2 T(n/2)+θ(n) >= O(n log n)
