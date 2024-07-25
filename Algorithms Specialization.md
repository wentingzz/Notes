# Algorithms

## Table of Contents
1. [Divide and Conquer, Sorting](#divide-and-conquer-sorting)
2. [Graph Search, Shortest Paths, and Data Structures](#graph-search-shortest-paths-and-data-structures)
3. [Greedy & DP](#greedy-algorithms--dynamic-programming): Scheduling, MST, Huffman; WIS, Knapsack, Sequence Alignment, Optimal BST

## Divide and Conquer, Sorting
<details>
  <summary>Divide & Conquer: Counting Inversions, Integer/Matrix Multiplication, Closest Pair, Master Method</summary>

  \
  **Counting Inversions**
  - Split into LHS and RHS, count inversion of each respectively, and count inversions between them
    ```
    def count_inversions(arr):
        if len(arr) < 2:
            return arr, 0
        mid = len(arr) // 2
        left, left_inv = count_inversions(arr[:mid])
        right, right_inv = count_inversions(arr[mid:])
        merged, split_inv = merge_and_count_split_inv(left, right)
        total_inv = left_inv + right_inv + split_inv
        return merged, total_inv
    
    def merge_and_count_split_inv(left, right):
        i = j = inv_count = 0
        merged = []
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                inv_count += len(left) - i
                j += 1
        merged.extend(left[i:])
        merged.extend(right[j:])
        return merged, inv_count
    ```
  
  **Integer Multiplication - Karatsuba**
  - Running time = $O(n^{log_2 3})$
  - Given two integers $x$ and $y$ to multiply, split each integer into two halves:
  $$x = x_1 \cdot 10^{\frac{n}{2}} + x_0$$
  $$y = y_1 \cdot 10^{\frac{n}{2}} + y_0$$
  - Recursively compute the following intermediate products:
       $$z_0 = x_0 y_0$$
       $$z_1 = (x_1 + x_0)(y_1 + y_0)$$
       $$z_2 = x_1 y_1 $$
  - The final product $z$ of $x$ and $y$ is given by:
       $$z = z_2 \cdot 10^n + (z_1 - z_2 - z_0) \cdot 10^{\frac{n}{2}} + z_0$$
    
    ```python
    def karatsuba(x, y):
        # Base case: if either x or y is a single-digit number, use simple multiplication
        if x < 10 or y < 10:
            return x * y
        
        # Calculate the number of digits in both x and y
        n = max(len(str(x)), len(str(y)))
        n2 = n // 2  # floor(n/2)
    
        # Split x and y into halves
        high1, low1 = x // 10**n2, x % 10**n2
        high2, low2 = y // 10**n2, y % 10**n2
    
        # Recursively calculate three products
        z0 = karatsuba(low1, low2)  # z0 = low1 * low2
        z1 = karatsuba((low1 + high1), (low2 + high2))  # z1 = (low1 + high1) * (low2 + high2)
        z2 = karatsuba(high1, high2)  # z2 = high1 * high2
    
        # Apply the Karatsuba formula to get the final result
        return z2 * 10**(2 * n2) + (z1 - z2 - z0) * 10**n2 + z0
    ```
  
  **Matrix Multiplication - Strassen's 7 Products**
  - Split each matrix $A$ and $B$ into four smaller matrices. If $A$ and $B$ are $n \times n$ matrices, split them into $\frac{n}{2} \times \frac{n}{2}$ submatrices.
  - Recursively compute the products of these smaller submatrices until the base case is reached (usually $1 \times 1$ matrices).
    ```math
    X = [[A B]    Y = [[E F]        X*Y = [[P5+P4-P2+P6  P1+P2]
        [C D]]        [G H]]              [P3+P4  P1+P5-P3-P7]]
    
    P1 = A(F-H)
    P2 = (A+B)H
    P3 = (C+D)E
    P4 = D(G-E)
    P5 = (A+D)(E+H)
    P6 = (B-D)(G+H)
    P7 = (A-C)(E+F)
    ```
  - Combine the results of smaller subproblems to obtain the final product matrix.
    ```
    def matrix_multiply_recursive(X, Y):
        n = len(X)
        if n == 1:
            return [[X[0][0] * Y[0][0]]]
    
        # Divide X and Y into quarters
        mid = n // 2
        A = [row[:mid] for row in X[:mid]]
        B = [row[mid:] for row in X[:mid]]
        C = [row[:mid] for row in X[mid:]]
        D = [row[mid:] for row in X[mid:]]
        E = [row[:mid] for row in Y[:mid]]
        F = [row[mid:] for row in Y[:mid]]
        G = [row[:mid] for row in Y[mid:]]
        H = [row[mid:] for row in Y[mid:]]
    
        # Recursive calls to compute the sub-products
        P1 = matrix_multiply_recursive(A, matrix_subtract(F, H))
        P2 = matrix_multiply_recursive(matrix_add(A, B), H)
        P3 = matrix_multiply_recursive(matrix_add(C, D), E)
        P4 = matrix_multiply_recursive(D, matrix_subtract(G, E))
        P5 = matrix_multiply_recursive(matrix_add(A, D), matrix_add(E, H))
        P6 = matrix_multiply_recursive(matrix_subtract(B, D), matrix_add(G, H))
        P7 = matrix_multiply_recursive(matrix_subtract(A, C), matrix_add(E, F))
    
        # Compute the sub-matrices of the result
        return C = [[P5+P4-P2+P6, P1+P2], [P3+P4, P1+P5-P3-P7]]
    ```
  
  **Closest Pair**
  - Sort points by x and y respectively,
  - Divide sorted points into half, and recursively find the closest pair in each half
  - Combine left_pair, right_pair, pair_in_between to find the best one. 
    ```
    # Recursive function to find closest pair
    def closest_pair_rec(P_x, P_y, n):
        # Base case: small n => directly compute
        if n <= 3:
            return brute_force_closest_pair(P_x)
        
        mid = n // 2
        Q_x = P_x[:mid]
        R_x = P_x[mid:]
        Q_y = [p for p in P_y if p in Q_x]
        R_y = [p for p in P_y if p in R_x]
        (p1, q1) = closest_pair_rec(Q_x, Q_y, mid)
        (p2, q2) = closest_pair_rec(R_x, R_y, n - mid)
        delta = min(dist(p1, q1), dist(p2, q2))
        (p3, q3) = closest_pair_split(P_x, P_y, delta, P_x[mid])
  
        return best((p1, q1), (p2, q2), (p3, q33))
    
    def closest_pair_split(P_x, P_y, delta, midpoint):
        best_strip_pair = None
        min_strip_distance = delta
        strip = [point for point in P_y if abs(point[0] - mid_point[0]) < delta]  # Scan the points near the midpoint to see if there is a better pair
        
        for i in range(len(strip)):
            for j in range(i + 1, len(strip)):
                if (strip[j][1] - strip[i][1]) >= min_strip_distance:
                    break
                elif dist(strip[i], strip[j]) < min_strip_distance:
                    min_strip_distance = dist(strip[i], strip[j])
                    best_strip_pair = (strip[i], strip[j])
        
        return best_strip_pair
        
    ```
  
  **Master Method**
  - The master theorem typically applies to recurrences of the form:
  
  $$T(n) = a T({\frac{n}{b}}) + O(n^d)$$
  
  - If $a = b^d$, time = $O(n^d log n)$
  - If $a < b^d$, time = $O(n^d)$
  - If $a > b^d$, time = $O(n^{log_b a})$

</details>

<details>
  <summary>Sort: Merge Sort, Quick Sort, Selection</summary>

  \
  **Merge Sort**
  - Split into LHS and RHS, sort them respectively, and merge them
  - Running time = $O(n \cdot log n)$
    ```python
    def merge_sort(arr):
        if len(arr) <= 1:
            return arr
        
        # Divide the array into two halves and recursively sort each half
        left_half = merge_sort(arr[:mid])
        right_half = merge_sort(arr[mid:])
        return merge(left_half, right_half)
    
    def merge(left, right):
        merged = []
        left_idx, right_idx = 0, 0
        
        # Merge elements from left and right into sorted order
        while left_idx < len(left) and right_idx < len(right):
            if left[left_idx] <= right[right_idx]:
                merged.append(left[left_idx])
                left_idx += 1
            else:
                merged.append(right[right_idx])
                right_idx += 1
        
        # Append remaining elements
        while left_idx < len(left):
            merged.append(left[left_idx])
            left_idx += 1
        while right_idx < len(right):
            merged.append(right[right_idx])
            right_idx += 1
        return merged
    ```
  
  **Quick Sort**
  - choose a pivot. Put all smaller elements on its left and all larger elements on its right
  - average running time = $O(n \cdot log n)$ and worst runninng time = $O(n^2)$
    ```python
    def partition(arr, low, high):
        pivot = arr[high]  # Choose the pivot element (last element in this case)
        i = low - 1  # Index of smaller element
        
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]  # Swap elements at i and j
        
        arr[i + 1], arr[high] = arr[high], arr[i + 1]  # Swap pivot with element at i + 1
        return i + 1  # Return the partition index
    
    def quick_sort_inplace(arr, low, high):
        if low < high:
            pi = partition(arr, low, high)  # Partition index
            quick_sort_inplace(arr, low, pi - 1)  # Sort left subarray
            quick_sort_inplace(arr, pi + 1, high)  # Sort right subarray
    ```
</details>

<details>
  <summary>Randomized Algorithms: Selection, Minimal Cut</summary>

  \
  **Selection**
  - Selecting the k-th smallest element
  - Average running time = $O(n)$, worst running time = $O(n^2)$
    ```python
    def quickselect(arr, low, high, k):
        if low < high:
            pi = partition(arr, low, high)
            if pi == k:
                return arr[pi]
            elif pi < k:
                return quickselect(arr, pi + 1, high, k)
            else:
                return quickselect(arr, low, pi - 1, k)
        return arr[low]
    ```
  
  **Minimal Cut using Random Contraction Algorithm**
  - cut-set = the bridging edges of divided sets A and B
  - $n = |V|$ and $m = |E|$
  - While there are more than 2 vertices in the graph:
    - Randomly select an edge $u, v ∈ E$
    - Merge (or contract) vertices $u$ and $v$ into a single vertex.
    - Update the edge set $E$ to remove self-loops but keep the multi-edges.
  
    ```python
    def random_contraction_algorithm(graph):
        while len(graph.vertices) > 2:
            # Randomly select an edge (u, v)
            u, v = random.choice(graph.edges)
            
            # Merge vertices u and v
            graph.contract(u, v)
        
        # The remaining edges form the cut
        min_cut = len(graph.edges)
        return min_cut
    
    def karger_min_cut(graph, num_iterations):
        min_cut = float('inf')
        for i in range(num_iterations):
            # Make a copy of the graph to avoid modifying the original
            temp_graph = copy.deepcopy(graph)
            cut = random_contraction_algorithm(temp_graph)
            if cut < min_cut:
                min_cut = cut
        return min_cut
    ```
</details>

## Graph Search, Shortest Paths, and Data Structures
<details>
  <summary>Graph Seach: BFS, DFS, Topological Sort, Strongly Connected Components (Kosaraju)</summary>

  \
  **BFS (queue)**
  - explore in layers
  - application:
    - shortest path between two points while all edges have the same weight
    - connectivity of undirected graph (if two vertices are connected) $O(|E|+|V|)$. To find all connected pieces, run BFS on all nodes if not visited.
    ```python
    def bfs(graph, start):
        visited = set()
        queue = Queue()
        queue.put(start)
        visited.add(start)
        
        while not queue.empty():
            vertex = queue.get()
            print(vertex, end=" ")
            
            for neighbor in graph[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.put(neighbor)
    ```
  
  **DFS**
  - $O(|E|+|V|)$
    ```python
    def dfs_recursive(graph, start, visited=None):
        if visited is None:
            visited = set()
        visited.add(start)
        
        print(start, end=' ')
        
        for neighbor in graph[start]:
            if neighbor not in visited:
                dfs_recursive(graph, neighbor, visited)
        return visited
    ```
  
  **Topological Sort**
  - directed graph
  - application
    - order to take course (prerequisite first)
    - compute strongly connected componenet (SCC) where there is a path from any vertex to every other vertex in the graph.
  - DFS implementation
    ```python
    def topological_sort_dfs(graph):
        visited = set()
        stack = []
    
        def dfs(node):
            visited.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor)
            stack.append(node)
    
        for node in graph:
            if node not in visited:
                dfs(node)
    
        stack.reverse()
        return stack
    ```
  - BFS with Kahn's algorithm (keep track of in_degree)
    ```python
    def topological_sort_kahns(graph):
        # Calculate in-degree of each node
        in_degree = {node: 0 for node in graph}
        for nodes in graph.values():
            for node in nodes:
                in_degree[node] += 1
    
        # Collect nodes with no incoming edges
        queue = deque([node for node in graph if in_degree[node] == 0])
        top_order = []
    
        while queue:
            node = queue.popleft()
            top_order.append(node)
    
            # Decrease the in-degree of neighboring nodes
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
    
        if len(top_order) == len(graph):
            return top_order
        else:
            raise ValueError("Graph has a cycle and cannot be topologically sorted")
    ```
  
  **Strongly Connected Components - Kosaraju's**
  - Perform a DFS on the transposed graph and keep track of the finish time of each vertex (push onto a stack when finished).
  - Perform a DFS on the original graph, in the order defined by the stack (highest finish time first). Each tree in this DFS is an SCC.
    ```python
    def kosaraju_scc(graph):
        def transpose(graph):
            transposed = {v: [] for v in graph}
            for v in graph:
                for neighbor in graph[v]:
                    transposed[neighbor].append(v)
            return transposed
        
        def dfs_first_pass(graph, v, visited, stack):
            visited[v] = True
            for neighbor in graph[v]:
                if not visited[neighbor]:
                    dfs_first_pass(graph, neighbor, visited, stack)
            stack.append(v)
        
        def dfs_second_pass(graph, v, visited, component):
            visited[v] = True
            component.append(v)
            for neighbor in graph[v]:
                if not visited[neighbor]:
                    dfs_second_pass(graph, neighbor, visited, component)
        
        # Step 1: Transpose the graph and DFS
        transposed_graph = transpose(graph)
        stack = []
        visited = {v: False for v in graph}
        for v in graph:
            if not visited[v]:
                dfs_first_pass(transposed_graph, v, visited, stack)
        
        # Step 2: Second DFS on the original graph
        visited = {v: False for v in graph}
        sccs = []
        while stack:
            v = stack.pop()
            if not visited[v]:
                component = []
                dfs_second_pass(graph, v, visited, component)
                sccs.append(component)
        
        return sccs
    ```


</details>

<details>
  <summary>Shortest Path: Dijkstra, Bellman-Ford, Floyd-Warshall</summary>

  \
  **Dijkstra**
  - we use bfs to compute shortest path on unweighted graph
  - Dijkstra can get shortest path on non-negative graph with time = $O((|V| + |E|) log |V|$ & space = $O(|V| + |E|)$
  - If graph has negative edges, Dijkstra's can compute a path but not guaranted to be the shortest.
  - If graph has negative cycles, special handling is needed to avoid inifinite loops in Dijkstra's.
    ```python
    def dijkstra(graph, start):
        # Priority queue to store (distance, node)
        pq = [(0, start)]
        # Dictionary to store the shortest path to each node
        distances = {node: float('inf') for node in graph}
        distances[start] = 0
        # Dictionary to store the path
        previous_nodes = {node: None for node in graph}
    
        while pq:
            current_distance, current_node = heapq.heappop(pq)
    
            # If the distance is greater than the recorded shortest distance, skip
            if current_distance > distances[current_node]:
                continue
    
            # Iterate through neighbors
            for neighbor, weight in graph[current_node].items():
                distance = current_distance + weight
    
                # Only consider this new path if it's better
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous_nodes[neighbor] = current_node
                    heapq.heappush(pq, (distance, neighbor))
    
        return distances, previous_nodes
    ```

  **Bellman-Ford**
  - Can handle graphs with negative edge weights and is capable of detecting negative weight cycles.
  - It is slower than Dijkstra's algorithm with a time complexity of $O(V⋅E)$. For dense graph, $E = O(V^2)$.
  - The vertice budget = 0, 1, 2, ..., # vertices - 2 (exclude start and end vertices)
  - \# candidate solutions = 1 + in_degree(destination vertex)
  - Application: internet routing
    ```python
    def bellman_ford(graph, source):
        # Step 1: Initialize distances
        distance = {v: float('inf') for v in graph.vertices}
        distance[source] = 0
    
        # Step 2: Relax edges |V| - 1 times
        for _ in range(len(graph.vertices) - 1):
            for u, v, w in graph.edges:
                distance[v] = min(distance[v], distance[u] + w)
    
        # Step 3: Check for negative weight cycles
        for u, v, w in graph.edges:
            if distance[u] + w < distance[v]:
                print("Graph contains a negative weight cycle")
    
        return distance
    ```
    
  **Floyd-Warshall**
  - All-Pairs Shortest Path (APSP)
  - Can handle graphs with negative edge weights and is capable of detecting negative weight cycles (`if dist[k][k] < 0`).
  - Run faster in dense graphs (# edges is close to the maximal # edges - $|E| ~ |V|(V – 1)/2$) with time complexity = $O(V^3)$
    ```python
    INF = float('inf')
    def floyd_warshall(graph):
        n = len(graph)
        dist = [[INF] * n for _ in range(n)]
        next_node = [[None] * n for _ in range(n)]
    
        # Initialize the distance matrix and next node matrix
        for i in range(n):
            for j in range(n):
                if i == j:
                    dist[i][j] = 0
                elif graph[i][j] != INF:
                    dist[i][j] = graph[i][j]
                    next_node[i][j] = j
    
        # Floyd-Warshall algorithm
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] != INF and dist[k][j] != INF and dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        next_node[i][j] = next_node[i][k]
    
        # Check for negative cycles
        for k in range(n):
            if dist[k][k] < 0:
                return "Graph contains a negative cycle"
    
        return dist, next_node
    ```

    **Johnson's Algorithm**
    - Uses a combination of reweighting (to eliminate negative weights) and then applying Dijkstra's algorithm for each vertex.
      - Adds one weight $p_v$ per vertex $v$. Then, new weight on edge $c'(s,e) = c(s,e) + p_s - p_e$
      - The middle weights are canceled out. $p_s$ and $p_v$ are the ones left.
      - Reweighting preserves shortest path
    - Uses one Bellman-Ford with Dijkstra's => $O(|E||V|log|V|)
      - Step 1: Adds one starting vertex $s$ where it connects to all vertices with weight 0
      - Step 2: Compute shortest path from $s$ to all other vertices using Bellman-Ford, the path cost = $p_v$ for each vertex
        - all path cost should be <= 0
        - if negative cycle is detected, it's not solvable
      - Step 3: Run Dijkstra’s Algorithm to every pair of vertices in reweighted graph
    ```python
    def johnsons_algorithm(graph):
        # Step 1
        new_vertex = 'q'
        new_edges = graph.edges + [(new_vertex, v, 0) for v in graph.vertices]
        new_graph = Graph(graph.vertices + [new_vertex], new_edges)

        # Step 2
        try:
            h = bellman_ford(new_graph, new_vertex)
        except ValueError as e:
            print(e)
            return None
        reweighted_edges = [(u, v, w + h[u] - h[v]) for u, v, w in graph.edges]
        reweighted_graph = Graph(graph.vertices, reweighted_edges)
        reweighted_graph.adjacency_list = defaultdict(list)
        
        #Step 3
        for u, v, w in reweighted_edges:
            reweighted_graph.adjacency_list[u].append((v, w))
    
        distances = {}
        for u in graph.vertices:
            dist = dijkstra(reweighted_graph, u)
            distances[u] = {v: dist[v] + h[v] - h[u] for v in graph.vertices}
        
        return distances
    ```
</details>

<details>
  <summary>Heaps: Binary Heap, K-ary Heap, Fibonacci Heap</summary>

  \
  **Binary Heap**
  - Application: Median Maintenance algorithm where we want medians of streamed numbers by having a min heap and a max heap where they are balanced 
  - Implement heap using a tree-like array where children is greater than parent
  - Insert and bubble-up if the new node's value is smaller than its parent. $O(log n)$
    ```c++
    void heapifyUp(int index) {
        int parent = (index - 1) / 2;
        while (index > 0 && heap[index] < heap[parent]) {
            swap(heap[index], heap[parent]);
            index = parent;
            parent = (index - 1) / 2;
        }
    }
  - Pop min, swap it with the last element, and bubble-down (swap with smaller child). $O(log n)$
    ```c++
    void heapifyDown(int index) {
        int left = 2 * index + 1, right = 2 * index + 2, smallest = index;

        if (left < heap.size() && heap[left] < heap[smallest])  smallest = left;
        if (right < heap.size() && heap[right] < heap[smallest]) smallest = right;

        if (smallest != index) {
            swap(heap[index], heap[smallest]);
            heapifyDown(smallest);
        }
    }
    int extractMin() {
        if (heap.empty())  throw runtime_error("Heap is empty");
        int minElement = heap[0];
        heap[0] = heap.back();
        heap.pop_back();
        heapifyDown(0);
        return minElement;
    }
    
    ```

  **K-ary Heap**
  - Children[A[i]]= [A[k\*i+1], ..., A[k\*i+k]]
  - Parent[A[i]] = A[floor($\frac{i-1}{k}$)]
  - Good for many Insert (bubble up is very similar to Binary Heap) and few ExtractMin
    ```
    void bubbleDown(int index) {
        while (true) {
            int minChildIndex = findMinChildIndex(index);
            if (minChildIndex == -1 || heap[minChildIndex] >= heap[index])
                break;
            swap(heap[index], heap[minChildIndex]);
            index = minChildIndex;
        }
    }
    
    int findMinChildIndex(int index) {
        int startChild = k * index + 1;
        if (startChild >= heap.size())
            return -1;
        int endChild = min(startChild + k, (int)heap.size());
        int minChildIndex = startChild;
        for (int i = startChild + 1; i < endChild; ++i) {
            if (heap[i] < heap[minChildIndex])
                minChildIndex = i;
        }
        return minChildIndex;
    }
    ```

**Fibonacci Heap**
- ExtractMin: return v pointered node `O(1)`
  - Merge children of delete node into list of roots
  ```

  ```
- Insert: `O(1)`
  - Create a new root node the item
  - Compare with previous best root, update if neccessary
- Merge two Fib heap: `O(1)`
  - Concatenate root lists by changing pointer of the double-linked list
  - Compare best roots to determine the new root of merged heap
- Decrease Priority
  - x is root ⇒ decrease its priority (we might need to check the min flag?)
  - x is not root ⇒ make it a root node first. x’s parent = p
    - if p is not root and flag is False ⇒ set it to True
    - if p is not root and flag is True ⇒ make p the root (we then look at p’s parent…)

    ```
    def promote(x):
        if x is not a root
            p = x.parent
            p.chlidren.remove(x)
            x.sibling.remove(x)
            remove x from sibling list
            roots.add(x)
            x.flag = False
            if p.flag: 
                promote(p)
            else if p is not a root:
                p.flag = True
    def decreasePriority(x):
        promote(x)
        compare x to best root, change if better
    ```
</details>

<details>
  <summary>Tree: BST, Red-Black Tree, AVL Tree</summary>

  \
  **Binary Search Tree**
  - MIN/MAX is found by going to left/right child until leaf is found
  - Pred/Succ
    - if left/right child exist, get the MAX/MIN of left/right leaf node
    - if not, find the first possible left/right parent (its right/left child = node's branch). 
    ```python
    function findPredecessor(node):
        if node.left != null:
            return findMax(node.left)
        else:
            parent = node.parent
            while parent != null and node == parent.left: # first possible "left parent"
                node = parent
                parent = parent.parent
            return parent
    
    function findMax(node):
        while node.right != null:
            node = node.right
        return node
    ```
  - Deletion
    - (0-1 child) Replace deleted node with its only child
    - (2 children) Swap deleted node with pred/succ
  - Select Problem (k smallest node) needs to keep track of size of the tree on each node
    - insertion/deletion will updates `size` for all parent nodes
    - selection
      ```python
      def kth_smallest(root, k):
          current = root
          while current:
              left_size = current.left.size if current.left else 0
              if k == left_size + 1:
                  return current.value
              elif k <= left_size:
                  current = current.left
              else:
                  k -= left_size + 1
                  current = current.right
          return None
      ```
  
  **Red-Black Tree**
  - Each node is either red or black. The root is black.
  - Red/Black Property: No red node can have a red parent. Consequently, every path from a node to its descendant leaves must have the same number of black nodes.
  - Leaf Nodes: All leaf nodes (NIL nodes or external nodes) are black.
  - Insertion Rules:
    - If the tree is empty, insert the new node as the root, coloring it black.
    - Insert the new node as a red node.
    - Rebalance the tree if necessary to maintain the Red-Black properties.
  - Deletion Rules:
    - Perform standard BST deletion.
    - If a black node is deleted, adjust the tree to preserve the Red-Black properties (rotation and recoloring if necessary).
  
  **AVL Tree**
  - `diff(height(n.left), height(n.right)) <= 1`
  - Roatation
    ```python
    def _right_rotate(self, y):
        x = y.left
        T2 = x.right
        # Perform rotation
        x.right = y
        y.left = T2
        return x
    def _left_rotate(self, x):
        y = x.right
        T2 = y.left
        # Perform rotation
        y.left = x
        x.right = T2
        return y
    ```
</details>

<details>
  <summary>Hashing: Linear Probing, Universial Hashing, Cuckcoo Hashing</summary>
  
  \
  **Simple Hash Functions**
  - Key-value structure. Good for insert/delete/lookup but not good for finding min/max/median
  - Load factor $\alpha = \frac{m}{n}$ where $m$ = # objects, $n$ = # buckets.
  - Rules of Thumb for **Choosing Buckets**
    - Use a prime number of buckets to avoid common factors with data.
    - Avoid primes close to $2^n$ or $10^n$ to prevent patterns based on data representation.
  - Division Metho: `hash_value = key % table_size`.
  - Multiplication Method: `hash_value = floor(a * table_size)` where `a` is in range of (0, 1)
  - **Double Hashing**: using a secondary hash function to calculate alternative hash locations for keys that collide.
  - **Hash Chaining**: hash_value -> a Linked_list to store all the values. $\alpha$ is possible to be greater than 1.
    - $O(1) ~ O(N)$ for insert/search/delete.

  
  **Linear Probing**
  - need $\alpha << 1$ to find open slot quickly
  - $O(1) ~ O(N)$ for insert/search/delete.
  - try `hash_value + probing_values` until an empty bucket is found.
    - quadratic probing: `probing_values = 1, 4, 9, ...`
    - double hashing: `probing_values = 1* h(k), 2*h(l), 3*h(k)...`
    ```
    def search(k):
        i = h(k)
        while H[i] is nonempty and contains a key != k	#infinite loop if full table
            i = (i + 1) mod N
        if H[i] is nonempty: return its value
        else: exception
    def set(k,v):
        i = h(k)
        while H[i] is nonempty and contains a key != k	#infinite loop
            i = (i + 1) mod N
        store (k,v) in H[i]
    ```
  - lazy delete
    - find k by v and mark the cell "nonempty but unused"
    - nonempty ⇒ search should not be stopped
    - unused ⇒ set can use
    - slow down followed search
  - eager delete
    ```
    def eager_delete(k):
        i = h(k)
        while H[i] is nonempty and contains a key 6= k
            i = (i + 1) mod N
        if H[i] is empty: exception
        j = (i + 1) mod N
        while H[j] is nonempty
            if h(H[j].key) is not in the (circular) range [i+1..j]: 
            #it was supposed to be at position before i+1
            # when mod, the range should be [i+1:end] union [0:j]
            #h(H[j].key) = hash the j’s key
              move H[j] to H[i]
              i = j
            j = j + 1
        clear H[i]
    ```
  
  **Cuckoo Hashing**
  - Two tables $H_0$  and $H_1$. load factor $α=n/N<1$
  - Two hash functions $h_0$  and $h_1$
  - Search(k): look in both places $H_0 [h_0 (k)]$, $H_1 [h_1 (k)]$
  - Delete(k): look at both tables and clear if found
  - Insert(k, v): store the pair in $H_0 [h_0 (k)]$
    - If pair $(k’, v’)$ exists, evict it and store in $H_1 [h_1 (k')]$
    - May require another pair (k’’, v’’) from $H_1 [h_1 (k')]$ and sore it in $H_0 [h_0 (k'')]$...
    - cycle ⇒ fail ⇒ rehash the entire table with different hash functions
    ```
    def insert(k,v):
        t = 0
        p = (k,v)
        repeat
            p ↔ H_t [h_t (k)]  #rotate t = 0 and t = 1
            if p is null: return
            if cycle detected:
                report failure / throw exception [rebuild table]
            t = 1 - t
            (k,v) = p
    ```
  
  
  **Bloom Filter**
  - Light-weighted hash table.
    - Each element maps to k-buckets in H. If all k are 1, this element exists. 
  - Fast insert and loopups. No deletion allowed
  - False positive rate (return TRUE when that value was not inserted)
  - Application: spellcheckers (didn't catch some typos), list of forbidden passwords (user can use some of the forbidden password), network routers (keep track of IP)
  ```
  
  
  ```
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
