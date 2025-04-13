import sys
import math
from collections import deque
import heapq

class Node:
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1
    
    def __lt__(self, other):
        return self.state < other.state  # For tie-breaking in priority queues

    def __repr__(self):
        return str(self.state)

class Graph:
    def __init__(self, filename):
        self.nodes = {}
        self.edges = {}
        self.origin = None
        self.destinations = []
        self.parse_file(filename)
    
    def parse_file(self, filename):
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
            
            section = None
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                if line.endswith(':'):
                    section = line[:-1]
                    continue
                
                if section == "Nodes":
                    # Parse node data: e.g., "1: (4,1)"
                    node_id, coords = line.split(':')
                    node_id = int(node_id.strip())
                    coords = coords.strip()[1:-1].split(',')
                    x, y = int(coords[0]), int(coords[1])
                    self.nodes[node_id] = (x, y)
                
                elif section == "Edges":
                    # Parse edge data: e.g., "(2,1): 4"
                    edge, cost = line.split(':')
                    edge = edge.strip()[1:-1].split(',')
                    from_node, to_node = int(edge[0]), int(edge[1])
                    cost = int(cost.strip())
                    
                    if from_node not in self.edges:
                        self.edges[from_node] = {}
                    self.edges[from_node][to_node] = cost
                
                elif section == "Origin":
                    self.origin = int(line.strip())
                
                elif section == "Destinations":
                    self.destinations = [int(dest.strip()) for dest in line.split(';')]
            
            print(f"Loaded graph with {len(self.nodes)} nodes, {sum(len(edges) for edges in self.edges.values())} edges")
            print(f"Origin: {self.origin}, Destinations: {self.destinations}")
        
        except Exception as e:
            print(f"Error parsing file: {e}")
            sys.exit(1)
    
    def get_neighbors(self, node_id):
        """Return list of neighboring nodes and costs in ascending order of node ID"""
        if node_id not in self.edges:
            return []
        return sorted([(to_node, cost) for to_node, cost in self.edges[node_id].items()])
    
    def is_goal(self, node_id):
        """Check if node is a destination"""
        return node_id in self.destinations
    
    def heuristic(self, node_id, dest_id=None):
        """Calculate Euclidean distance heuristic"""
        if dest_id is None:
            # If no specific destination provided, use minimum distance to any destination
            return min(self.heuristic(node_id, dest) for dest in self.destinations)
        
        x1, y1 = self.nodes[node_id]
        x2, y2 = self.nodes[dest_id]
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

class SearchAlgorithm:
    def __init__(self, graph):
        self.graph = graph
        self.nodes_created = 0
    
    def solution_path(self, node):
        """Return the sequence of actions and states from the root to this node"""
        path = []
        current = node
        while current:
            path.append(current.state)
            current = current.parent
        return path[::-1]  # Reverse to get from start to goal

    def output_solution(self, filename, method, goal_node):
        """Output solution in required format"""
        path = self.solution_path(goal_node)
        
        print(f"{filename} {method}")
        print(f"Goal: {goal_node.state} | Total Nodes: {self.nodes_created} | Cost: {goal_node.path_cost}")
        print(f"Path: {' + '.join(map(str, path))}")

class DFS(SearchAlgorithm):
    """Depth-First Search"""
    def search(self):
        # Initialize
        start_node = Node(self.graph.origin)
        self.nodes_created = 1
        
        if self.graph.is_goal(start_node.state):
            return start_node
        
        # Use a stack for DFS
        stack = [start_node]
        explored = set()
        
        while stack:
            # Pop from the end of the list (stack behavior)
            node = stack.pop()
            
            if node.state in explored:
                continue
                
            explored.add(node.state)
            
            # Expand the node
            neighbors = self.graph.get_neighbors(node.state)
            
            # Process neighbors in reverse order to expand lower-numbered nodes first
            for neighbor_state, step_cost in reversed(neighbors):
                if neighbor_state not in explored:
                    child = Node(
                        state=neighbor_state,
                        parent=node,
                        action=f"{node.state}->{neighbor_state}",
                        path_cost=node.path_cost + step_cost
                    )
                    self.nodes_created += 1
                    
                    if self.graph.is_goal(child.state):
                        return child
                    
                    stack.append(child)
        
        return None  # No solution found

class BFS(SearchAlgorithm):
    """Breadth-First Search"""
    def search(self):
        # Initialize
        start_node = Node(self.graph.origin)
        self.nodes_created = 1
        
        if self.graph.is_goal(start_node.state):
            return start_node
        
        # Use a queue for BFS
        queue = deque([start_node])
        explored = set([start_node.state])  # Track explored states to avoid duplicates
        
        while queue:
            # Pop from the beginning of the queue (FIFO)
            node = queue.popleft()
            
            # Expand the node
            neighbors = self.graph.get_neighbors(node.state)
            
            # Process neighbors in ascending order
            for neighbor_state, step_cost in neighbors:
                if neighbor_state not in explored:
                    child = Node(
                        state=neighbor_state,
                        parent=node,
                        action=f"{node.state}->{neighbor_state}",
                        path_cost=node.path_cost + step_cost
                    )
                    self.nodes_created += 1
                    
                    if self.graph.is_goal(child.state):
                        return child
                    
                    explored.add(neighbor_state)  # Mark as explored when added to queue
                    queue.append(child)
        
        return None  # No solution found

class GBFS(SearchAlgorithm):
    """Greedy Best-First Search"""
    def search(self):
        # Initialize
        start_node = Node(self.graph.origin)
        self.nodes_created = 1
        
        if self.graph.is_goal(start_node.state):
            return start_node
        
        # Use a priority queue for GBFS
        frontier = []
        heapq.heappush(frontier, (self.graph.heuristic(start_node.state), start_node))
        explored = set()
        
        while frontier:
            _, node = heapq.heappop(frontier)
            
            if self.graph.is_goal(node.state):
                return node
                
            if node.state in explored:
                continue
                
            explored.add(node.state)
            
            # Expand the node
            neighbors = self.graph.get_neighbors(node.state)
            
            # Process neighbors in ascending order
            for neighbor_state, step_cost in neighbors:
                if neighbor_state not in explored:
                    child = Node(
                        state=neighbor_state,
                        parent=node,
                        action=f"{node.state}->{neighbor_state}",
                        path_cost=node.path_cost + step_cost
                    )
                    self.nodes_created += 1
                    
                    # Add child to frontier with heuristic as priority
                    h = self.graph.heuristic(child.state)
                    heapq.heappush(frontier, (h, child))
        
        return None  # No solution found

class AS(SearchAlgorithm):
    """A* Search"""
    def search(self):
        # Initialize
        start_node = Node(self.graph.origin)
        self.nodes_created = 1
        
        if self.graph.is_goal(start_node.state):
            return start_node
        
        # Use a priority queue for A*
        frontier = []
        heapq.heappush(frontier, (start_node.path_cost + self.graph.heuristic(start_node.state), start_node))
        explored = {}
        
        while frontier:
            _, node = heapq.heappop(frontier)
            
            if self.graph.is_goal(node.state):
                return node
            
            # Only process if this is the cheapest path to this state
            if node.state in explored and explored[node.state] <= node.path_cost:
                continue
                
            explored[node.state] = node.path_cost
            
            # Expand the node
            neighbors = self.graph.get_neighbors(node.state)
            
            # Process neighbors in ascending order
            for neighbor_state, step_cost in neighbors:
                child_cost = node.path_cost + step_cost
                
                # Skip if we've found a better path to this neighbor
                if neighbor_state in explored and explored[neighbor_state] <= child_cost:
                    continue
                    
                child = Node(
                    state=neighbor_state,
                    parent=node,
                    action=f"{node.state}->{neighbor_state}",
                    path_cost=child_cost
                )
                self.nodes_created += 1
                
                # Add child to frontier with f(n) = g(n) + h(n) as priority
                h = self.graph.heuristic(child.state)
                f = child_cost + h
                heapq.heappush(frontier, (f, child))
        
        return None  # No solution found

class CUS1(SearchAlgorithm):
    """Custom Search 1: Dijkstra's Algorithm"""
    def search(self):
        # Initialize
        start_node = Node(self.graph.origin)
        self.nodes_created = 1
        
        if self.graph.is_goal(start_node.state):
            return start_node
        
        # Priority queue using path cost
        frontier = []
        heapq.heappush(frontier, (start_node.path_cost, start_node))
        
        # Keep track of the best known costs to reach each node
        best_costs = {self.graph.origin: 0}
        
        while frontier:
            current_cost, node = heapq.heappop(frontier)
            
            # Skip if we've found a better path to this node already
            if current_cost > best_costs.get(node.state, float('inf')):
                continue
                
            if self.graph.is_goal(node.state):
                return node
                
            # Expand the node
            neighbors = self.graph.get_neighbors(node.state)
            
            for neighbor_state, step_cost in neighbors:
                new_cost = node.path_cost + step_cost
                
                # If this is the best path to neighbor_state we've found so far
                if new_cost < best_costs.get(neighbor_state, float('inf')):
                    best_costs[neighbor_state] = new_cost
                    child = Node(
                        state=neighbor_state,
                        parent=node,
                        action=f"{node.state}->{neighbor_state}",
                        path_cost=new_cost
                    )
                    self.nodes_created += 1
                    heapq.heappush(frontier, (new_cost, child))
        
        return None  # No solution found

class CUS2(SearchAlgorithm):
    """Custom Search 2: Iterative Deepening A* (IDA*)"""
    def search(self):
        # Initialize
        start_node = Node(self.graph.origin)
        self.nodes_created = 1
        
        if self.graph.is_goal(start_node.state):
            return start_node
        
        def search(path, g, bound):
            node = path[-1]
            f = g + self.graph.heuristic(node.state)
            
            if f > bound:
                return f, None
                
            if self.graph.is_goal(node.state):
                return f, node
                
            min_f = float('inf')
            best_node = None
            
            # Expand the node
            neighbors = self.graph.get_neighbors(node.state)
            
            for neighbor_state, step_cost in neighbors:
                # Skip if already in path (to avoid cycles)
                if any(n.state == neighbor_state for n in path):
                    continue
                    
                child = Node(
                    state=neighbor_state,
                    parent=node,
                    action=f"{node.state}->{neighbor_state}",
                    path_cost=node.path_cost + step_cost
                )
                self.nodes_created += 1
                
                new_path = path + [child]
                new_f, result = search(new_path, g + step_cost, bound)
                
                if result is not None:
                    return new_f, result
                    
                if new_f < min_f:
                    min_f = new_f
                    
            return min_f, best_node
        
        bound = self.graph.heuristic(start_node.state)
        path = [start_node]
        
        while True:
            new_bound, result = search(path, 0, bound)
            
            if result is not None:
                return result
                
            if new_bound == float('inf'):
                return None
                
            bound = new_bound

def run_search(filename, method):
    graph = Graph(filename)
    
    if method == "DFS":
        algorithm = DFS(graph)
    elif method == "BFS":
        algorithm = BFS(graph)
    elif method == "GBFS":
        algorithm = GBFS(graph)
    elif method == "AS":
        algorithm = AS(graph)
    elif method == "CUS1":
        algorithm = CUS1(graph)
    elif method == "CUS2":
        algorithm = CUS2(graph)
    else:
        print(f"Unknown method: {method}")
        sys.exit(1)
    
    print(f"Running {method} on {filename}...")
    result = algorithm.search()
    
    if result:
        algorithm.output_solution(filename, method, result)
    else:
        print(f"No solution found using {method} on {filename}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python search.py <filename> <method>")
        print("Methods: DFS, BFS, GBFS, AS, CUS1, CUS2")
        sys.exit(1)
    
    filename = sys.argv[1]
    method = sys.argv[2].upper()
    
    run_search(filename, method)

if __name__ == "__main__":
    main()