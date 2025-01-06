Robotics Fundations:
* C-space (configuration space) = space of all possible state that robot can attain
* Robot can move from state i to state j only if state i and j are in the connected componenet inside the C-Space representation

[A* search](https://github.com/wentingzz/Notes/blob/main/MCS271%20AI%20Notes.md#heuristic-search-a-best-first-local)
- `esti_final_cost(n)` = `cost_to_cur(n)` + `esti_cost_to_goal(n)` (Make sure `esti_cost_to_goal(n) < esti_cost_to_goal(n)`)
- `cost_to_cur` is infinity if the node is not reached yet
- Use minheap (sort by `esti_final_cost(n)`) to determine the next node
  - When a node is popped from the minheap, we visit its unvisited neighbor.
  - For each neighbor, update `cost_to_cur`, `esti_final_cost`, `parent_node` if we find a better path
  - If `esti_final_cost > cost_to_cur(final)`, we add this neighbor to minheap
- Note:
  - Dijkstra's (`cost_to_cur(n) = esti_final_cost(n)`)
  - We only underestimate the cost to goal and never overestimate (Otherwise, the optimal may not be correctly sorted in the heap and never be processed. It will return non-optimal solution)



