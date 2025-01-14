Robotics Fundations:
* C-space (configuration space) = space of all possible state that robot can attain
* Robot can move from state i to state j only if state i and j are in the connected componenet inside the C-Space representation

[A* Search](https://github.com/wentingzz/Notes/blob/main/MCS271%20AI%20Notes.md#heuristic-search-a-best-first-local)
- `esti_final_cost(n)` = `cost_to_cur(n)` + `esti_cost_to_goal(n)` (Make sure `esti_cost_to_goal(n) < esti_cost_to_goal(n)`)
- `cost_to_cur` is infinity if the node is not reached yet
- Use minheap (sort by `esti_final_cost(n)`) to determine the next node
  - When a node is popped from the minheap, we visit its unvisited neighbor.
  - For each neighbor, update `cost_to_cur`, `esti_final_cost`, `parent_node` if we find a better path
  - If `esti_final_cost > cost_to_cur(final)`, we add this neighbor to minheap
- Note:
  - Dijkstra's (`cost_to_cur(n) = esti_final_cost(n)`) is useful when we only know the edges without nodes position.
  - If we know coordinates of the nodes, some typical heuristic functions can be
    - Manhattan Distance (grid-based problems + four directions) = `dx + dy`.
    - Euclidean Distance (diagonal movement) = `sqrt((dx)² + (dy)²))`
    - Octile Distance (8-directional movement) = `max(dx, dy) + (√2 - 1) * min(dx, dy)`
  - We only **underestimate** the cost to goal and never overestimate (Otherwise, the optimal may not be correctly sorted in the heap and never be processed. It will return non-optimal solution)
- <details>
  <summary>Code</summary>
  
  ```
  vector<int> buildPath(vector<int> parent, int node){
      vector<int> res;
      res.push_back(node);
      while(parent[node] != -1){
          node = parent[node];
          res.push_back(node);
      }
      
      reverse(res.begin(), res.end());
      for(int& n: res) cout << n << " ";
      cout << endl;
      return res;
  }
  
  vector<int> ASearch(int n, vector<vector<int>> edges, int start, int goal){ //[n1, n2, weight]
      int minEdge = INT_MAX;
      vector<vector<pair<int, int>>> adj(n); //adjacent list {neighbor, weight}
      for(auto e: edges){
          adj[e[0]].push_back({e[1], e[2]});
          adj[e[1]].push_back({e[0], e[2]});
          minEdge = min(minEdge, e[2]);
      }
      vector<int> parent(n, -1);
      vector<int> dist(n, INT_MAX); //distance from start to node
      vector<int> esti(n, INT_MAX); //estimate cost to goal
      dist[start] = 0, esti[start] = minEdge;
      
      priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
      pq.push({esti[start], start});
      while(!pq.empty()){
          auto [w, cur] = pq.top(); pq.pop();
          for(auto [next, cost]: adj[cur]){
              if(dist[next] < dist[cur] + cost) continue;
              parent[next] = cur;
              dist[next] = dist[cur] + cost;
              esti[next] = dist[next] + minEdge;
              if(dist[goal] < esti[next]) continue;
              pq.push({esti[next], next});
          }
      }
      
      return buildPath(parent, goal);
  }
  
  int main() {
      vector<vector<int>> edges = {{0,2,18}, {0,3,12}, {0,4,30},
                                   {2,1,27}, {2,5,15}, {1,5,10},
                                   {3,5,20}, {3,4,8}, {4,5,10}};
      ASearch(6, edges, 0, 5);
      return 0;
  }
  ```
</details>

- Grid Representation: Divide each of the `n` dimensions into `k` spaces => `n^k` cells of C-space representation. Costly if `n` or `k` grows.
- Multi-Resolution Grid: Grow collision node to get fineer resolution. Free space will be blurry resolution. Less costly
- PRM (Probablistic RoadMap): randomly select free space, connect nearby neighbors if no collision, connect start_node, goal_node to the graph (no collision). Good for multi-query (one graph for multiple starts/goals)
  - sampling over uniform/un-uniform distribution over C space.
  - sampling by deterministic algorithm (Van der Corput (binary-search-tree-like), Halton sequence (
- Randomly Exploring Random Tree (RRT): randomly sample a new node `n_samp`, find the nearst neighbor `n_neig` and connect it to `n_new` (`n_new` is between `n_samp` and `n_neig` but has no collision. If there is collision, discard this sample), do so until `n_new` is inside goal space. Better than Random Walk because it explores more in C space.
- Attractive/Repulsive Potential Obstacle (may stuck in local minimums)
  - Command robot forces equal to the negative of the gradient of P, plus some damping forces (to reduce or eliminate oscillation of the robot about the goal configuration).
  - Command robot velocities equal to the negative of the gradient of P (Under this velocity control law, the robot is treated as a kinematic system (not a second-order dynamic system), and there is no oscillation).
- To minimize cost of motion:
  - Shooting: try motion, adjust collision points and retry until it's collision free and minimize the cost.
    - The gradient of a representation of the trajectory with respect to the controls (sensitivity of the trajectory with respect to the controls)
    - The gradient of the constraints with respect to the trajectory (sensitivity of the constraints with respect to the trajectory)
    - The gradient of the cost with respect to the controls (change the controls to reduce the cost)
  - Collocation:
  - Transcription:  

Control System
- `e_ss` stable-state error response (accepatable final stable state with some error) (`e_ss` = 0 is better)
- transient error response: 
  - `overshoot` negative (`theta` > `theta_d`) difference between state and `e_ss` (no overshoot is better)
  - `settle_time` time to settle in range of \[0, `e_ss`\] (no settle_time is better)
- Linear Error Dynmaics (can be homogeneous or nonhomogeneous)
  - like mass-spring-damper. `theta_e` (error dynamics) = position of the mass `m` = linear ordinary differential equation = `m * theta_e'' + b * theta_e ' + k * theta_e`
  - Designing a controller = choosing the spring constant `k` and the damping constant `b`.
    - Large `k` => spring pulls the error toward zero faster
    - Large `b` => less overshoot and oscillation
  - In pth-order linear differential equation, A necessary condition for stability is that all coefficients `a_i'` be positive. The real components of all the eigenvalues must be negative.
  - Standard First-Order equation `theta_e(t)' + theta_e(t) / t = 0` where `t = b/k` is stable if `t > 0`
    - time constant `t` of the decay = `1/a`
    - Small `t` (spring gets stiffer or the damper gets softer) => fast decay (steady-state error = 0, overshoot = 0)
    - 2% settling time (time for the error to decay to 2 percent of its initial value) = `4k`
  - Standard Second-Order Equation `theta_e(t)'' + 2*c*w * theta_e(t)' + w*w + theta_e(t) = 0` where natural frequency `w = sqrt(k/m)`, damping ratio `c = b/(2*sqrt(k*m))`
    - `c > 1` two solution overdamped. (larger solution => fast settling time, positive solution => unstable)
    - `c = 1` one solution critically damped. 2% settling time = 4/w
    - `c < 1` complex number solution underdamped. 2% settling time = 4/w. Incresing overshoot


