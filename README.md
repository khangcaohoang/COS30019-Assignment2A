# Assignment 2 A
Pathfinding algorithms.

## Requirements
**NOTE 1:** The objective is to reach one of the destination nodes.

**NOTE 2:** *When all else is equal*, nodes should be expanded according to the ascending order, i.e., from the smaller to the bigger nodes. For instance, when all else are being equal between nodes 4 and 7, Node 4 should be expanded before Node 7. Furthermore, *when all else is equal*, the two nodes N 1 and N 2 *on two different branches* of the search tree should be expanded according to the chronological order: if node N 1 is added BEFORE node N 2 then N 1 is expanded BEFORE node N 2.

Needs to be called like: *python search.py \<filename> \<method>*

| Output should be: | |
| - | - |
| filename | method |
| goal | number_of_nodes |
| path |

### Required strategies
| Search Strategy | Description | Method |
| -- | -- | -- |
| Uninformed |
| depth-first search | Select one option, try it, go back when there are no more options | DFS |
| breadth-first search | Expand all options one level at a time | BFS |
| Informed |
| greedy best-first | Use only the cost to reach the goal from the current node to evaluate the node | GBFS |
| A* (“A Star”) | Use both the cost to reach the goal from the current node and the cost to reach this node to evaluate the node | AS |
| Custom |
| Your search strategy 1 | An uninformed method to find a path to reach the goal. | CUS1 |
| Your search strategy 2 | An informed method to find a shortest path (with least moves) to reach the goal. | CUS2 |

## TODO
- [X] Add full path output to alogrithms.
- [X] Implement custom uninformed algorithm.
- [ ] Implement custom informed algorithm.
- [ ] Import text file.
