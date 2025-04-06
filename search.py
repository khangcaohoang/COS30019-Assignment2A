
from collections import deque
import sys
import functools
import heapq
import numpy as np

def is_in(elt, seq):
	
	return any(x is elt for x in seq)

def distance(a, b):
	
	xA, yA = a
	xB, yB = b
	return np.hypot((xA - xB), (yA - yB))

def memoize(fn, slot=None, maxsize=32):
	
	# BUG Trying to use the slot causes a TypeError. Unable to fix it.
	if slot:
		def memoized_normal(obj, *args):
			if hasattr(obj, slot):
				return getattr(obj, slot)
			else:
				val = fn(obj, *args)
				setattr(obj, slot, val)
				return val
		out = memoized_normal
	else:
		@functools.lru_cache(maxsize=maxsize)
		def memoized_cached(*args):
			return fn(*args)
		out = memoized_cached

	return out

class PriorityQueue:


	def __init__(self, order='min', f=lambda x: x):
		self.heap = []
		if order == 'min':
			self.f = f
		elif order == 'max':  # now item with max f(x)
			self.f = lambda x: -f(x)  # will be popped first
		else:
			raise ValueError("Order must be either 'min' or 'max'.")

	def append(self, item):
		"""Insert item at its correct position."""
		heapq.heappush(self.heap, (self.f(item), item))

	def extend(self, items):
		"""Insert each item in items at its correct position."""
		for item in items:
			self.append(item)

	def pop(self):
		
		if self.heap:
			return heapq.heappop(self.heap)[1]
		else:
			raise Exception('Trying to pop from empty PriorityQueue.')

	def __len__(self):
		"""Return current capacity of PriorityQueue."""
		return len(self.heap)

	def __contains__(self, key):
		"""Return True if the key is in PriorityQueue."""
		return any([item == key for _, item in self.heap])

	def __getitem__(self, key):
		
		for value, item in self.heap:
			if item == key:
				return value
		raise KeyError(str(key) + " is not in the priority queue")

	def __delitem__(self, key):
		"""Delete the first occurrence of key."""
		try:
			del self.heap[[item == key for _, item in self.heap].index(True)]
		except ValueError:
			raise KeyError(str(key) + " is not in the priority queue")
		heapq.heapify(self.heap)

class Problem:
	

	def __init__(self, initial, goal=None):
		
		self.initial = initial
		self.goal = goal

	def actions(self, state):
		
		raise NotImplementedError

	def result(self, state, action):
		
		raise NotImplementedError

	def goal_test(self, state):
		
		if isinstance(self.goal, list):
			return is_in(state, self.goal)
		else:
			return state == self.goal

	def path_cost(self, cost, state1, action, state2):
		
		return cost + 1

	def value(self, state):
		
		raise NotImplementedError

class Node:


	f = None

	def __init__(self, state, parent=None, action=None, path_cost=0):
		
		self.state = state
		self.parent = parent
		self.action = action
		self.path_cost = path_cost
		self.depth = 0
		if parent:
			self.depth = parent.depth + 1

	def __repr__(self):
		return "<Node {}>".format(self.state)

	def __lt__(self, node):
		return self.state < node.state

	def expand(self, problem):
		
		return [self.child_node(problem, action)
				for action in problem.actions(self.state)]

	def child_node(self, problem, action):
		
		next_state = problem.result(self.state, action)
		next_node = Node(next_state, self, action, problem.path_cost(self.path_cost, self.state, action, next_state))
		return next_node

	def solution(self):
		
		return [node.action for node in self.path()[1:]]

	def path(self):
		
		node, path_back = self, []
		while node:
			path_back.append(node)
			node = node.parent
		return list(reversed(path_back))

	# We want for a queue of nodes in breadth_first_graph_search or
	# astar_search to have no duplicated states, so we treat nodes
	# with the same state as equal.

	def __eq__(self, other):
		return isinstance(other, Node) and self.state == other.state

	def __hash__(self):
		
		return hash(self.state)

class Graph:

	def __init__(self, graph_dict=None, directed=True):
		self.locations = {}
		self.least_costs = {}
		self.graph_dict = graph_dict or {}
		self.directed = directed
		if not directed:
			self.make_undirected()

	def make_undirected(self):
		
		for a in list(self.graph_dict.keys()):
			for (b, dist) in self.graph_dict[a].items():
				self.connect1(b, a, dist)

	def connect(self, A, B, distance=1):
		"""Add a link from A and B of given distance, and also add the inverse
		link if the graph is undirected."""
		self.connect1(A, B, distance)
		if not self.directed:
			self.connect1(B, A, distance)

	def connect1(self, A, B, distance):
		"""Add a link from A to B of given distance, in one direction only."""
		self.graph_dict.setdefault(A, {})[B] = distance

	def get(self, a, b=None):
		"""Return a link distance or a dict of {node: distance} entries.
		.get(a,b) returns the distance or None;
		.get(a) returns a dict of {node: distance} entries, possibly {}."""
		links = self.graph_dict.setdefault(a, {})
		if b is None:
			return links
		else:
			return links.get(b)

	def nodes(self):
		"""Return a list of nodes in the graph."""
		s1 = set([k for k in self.graph_dict.keys()])
		s2 = set([k2 for v in self.graph_dict.values() for k2, v2 in v.items()])
		nodes = s1.union(s2)
		return list(nodes)

class GraphProblem(Problem):
	"""The problem of searching a graph from one node to another."""

	def __init__(self, initial, goal, graph):
		super().__init__(initial, goal)
		self.graph = graph

	def actions(self, state):
		"""The actions at a graph node are just its neighbors."""
		return list(self.graph.get(state).keys())

	def result(self, state, action):
		"""The result of going to a neighbor is just that neighbor."""
		return action

	def path_cost(self, cost, state1, action, state2):
		"""Cost is the cost so far."""
		return cost + (self.graph.get(state1, state2) or np.inf)

	def find_min_edge(self):
		"""Find minimum value of edges."""
		m = np.inf
		for d in self.graph.graph_dict.values():
			local_min = min(d.values())
			m = min(m, local_min)

		return m

	def h(self, node):
		
		#locs = getattr(self.graph, 'locations', None)
		locs = self.graph.locations
		if locs:
			if type(node) is str:
				return int(distance(locs[node], locs[self.goal]))

			return int(distance(locs[node.state], locs[self.goal]))
		else:
			return np.inf

def depth_first_graph_search(problem):

	frontier = [(Node(problem.initial))]  # Stack

	explored = set()
	while frontier:
		node = frontier.pop()
		if problem.goal_test(node.state):
			return node
		explored.add(node.state)
		frontier.extend(child for child in node.expand(problem)
						if child.state not in explored and child not in frontier)
	return None

def breadth_first_graph_search(problem):
	
	node = Node(problem.initial)
	if problem.goal_test(node.state):
		return node
	frontier = deque([node])
	explored = set()
	while frontier:
		node = frontier.popleft()
		explored.add(node.state)
		for child in node.expand(problem):
			if child.state not in explored and child not in frontier:
				if problem.goal_test(child.state):
					return child
				frontier.append(child)
	return None

def best_first_graph_search(problem, f, display=False):
	
	#f = memoize(f, 'f')
	f = memoize(f)
	node = Node(problem.initial)
	frontier = PriorityQueue('min', f)
	frontier.append(node)
	explored = set()
	while frontier:
		node = frontier.pop()
		if problem.goal_test(node.state):
			if display:
				print(len(explored), "paths have been expanded and", len(frontier), "paths remain in the frontier")
			return node
		explored.add(node.state)
		for child in node.expand(problem):
			if child.state not in explored and child not in frontier:
				frontier.append(child)
			elif child in frontier:
				if f(child) < frontier[child]:
					del frontier[child]
					frontier.append(child)
	return None

# This is effectively the greedy best first search
def uniform_cost_search(problem, display=False):
	"""[Figure 3.14]"""
	return best_first_graph_search(problem, lambda node: node.path_cost, display)

# Is this needed?
def depth_limited_search(problem, limit=50):
	"""[Figure 3.17]"""

	def recursive_dls(node, problem, limit):
		if problem.goal_test(node.state):
			return node
		elif limit == 0:
			return 'cutoff'
		else:
			cutoff_occurred = False
			for child in node.expand(problem):
					result = recursive_dls(child, problem, limit - 1)
					if result == 'cutoff':
						cutoff_occurred = True
					elif result is not None:
						return result
			return 'cutoff' if cutoff_occurred else None

	# Body of depth_limited_search:
	return recursive_dls(Node(problem.initial), problem, limit)

# Is this needed?
def iterative_deepening_search(problem):
	
	for depth in range(sys.maxsize):
		result = depth_limited_search(problem, depth)
		if result != 'cutoff':
			return result

def astar_search(problem, h=None, display=False):
	
	#h = memoize(h or problem.h, 'h')
	h = memoize(h or problem.h)
	return best_first_graph_search(problem, lambda n: n.path_cost + h(n), display)

def dijkstra_search(problem, display=False):
    """CUS1: Dijkstra's algorithm for finding the shortest path."""
    # Initialize distances and predecessors
    distances = {problem.initial: 0}
    predecessors = {problem.initial: None}
    visited_nodes = 0
    
    # Priority queue using path cost
    frontier = PriorityQueue('min', lambda node: node.path_cost)
    frontier.append(Node(problem.initial))
    
    explored = set()
    
    while frontier:
        node = frontier.pop()
        visited_nodes += 1
        
        if problem.goal_test(node.state):
            if display:
                print(len(explored), "paths have been expanded and", len(frontier), "paths remain in the frontier")
            return node, visited_nodes
            
        explored.add(node.state)
        
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                distances[child.state] = child.path_cost
                predecessors[child.state] = node.state
                frontier.append(child)
            elif child in frontier:
                if child.path_cost < distances[child.state]:
                    distances[child.state] = child.path_cost
                    predecessors[child.state] = node.state
                    del frontier[child]
                    frontier.append(child)
    
    return None, visited_nodes

def CUS1(problem, display=False):
    """Custom Search 1: Dijkstra's algorithm implementation."""
    node, visited_nodes = dijkstra_search(problem, display)
    if node:
        return node
    return None

def ida_star_search(problem, h=None, display=False):
    """CUS2: Iterative Deepening A* search."""
    h = memoize(h or problem.h)
    
    def search(path, g, bound, visited):
        node = path[-1]
        f = g + h(node)
        if f > bound:
            return f, visited
        if problem.goal_test(node.state):
            return (path, visited)
        
        min_cost = float('inf')
        for child in node.expand(problem):
            if child not in path:
                visited += 1
                path.append(child)
                result, visited = search(path, g + problem.path_cost(g, node.state, None, child.state), bound, visited)
                if isinstance(result, list):
                    return (result, visited)
                if result < min_cost:
                    min_cost = result
                path.pop()
        return min_cost, visited
    
    bound = h(Node(problem.initial))
    path = [Node(problem.initial)]
    visited = 0
    
    while True:
        result, visited = search(path, 0, bound, visited)
        if isinstance(result, list):
            return result[-1], visited
        if result == float('inf'):
            return None, visited
        bound = result

def CUS2(problem, h=None, display=False):
    """Custom Search 2: IDA* implementation."""
    node, visited_nodes = ida_star_search(problem, h, display)
    if node:
        return node
    return None
def import_graph(_file: str):
	"""Import the graph data. Create the GraphProblem and return it."""''
	class importer:
		nodes = []
		edges = []
		initial = None
		goal = None

	file_import = importer()
	graph = Graph()
	GraphProblem(file_import.initial, file_import.goal, graph)

def create_graph():
	graph = Graph(dict(
		A=dict(C=5, D=6),
		B=dict(A=4, C=4),
		C=dict(A=5, E=6, F=7),
		D=dict(A=6, C=5, E=7),
		E=dict(C=6, D=8, G=6),
		F=dict(C=7, G=6),
		G=dict(E=6, F=6)
	))
	graph.locations = dict(
		A=(4,1),
		B=(2,2),
		C=(4,4),
		D=(6,3),
		E=(5,6),
		F=(7,5),
		G=(7,7)
	)
	initial = "G"
	goal = "A"
	problem = GraphProblem(initial, goal, graph)
	return problem, goal

def select_method(_method: str):
	
	match _method:
		case "DFS":
			return depth_first_graph_search
		case "BFS":
			return breadth_first_graph_search
		case "GBFS":
			return uniform_cost_search
		case "AS":
			return astar_search
		case "IDS":
			return iterative_deepening_search
		case "CUS1":
			return CUS1
		case "CUS2":
			return CUS2
		case _:
			return None

if __name__ == "__main__":

	if len(sys.argv) < 3:
		print("Missing arguments: python search.py <filename> <method>")
		quit()

	if len(sys.argv) > 3:
		print("Excess arguments: python search.py <filename> <method>")

	# Extract parameter 1: filename of graph
	#graph_result = import_graph(sys.argv[1])
	#if graph_result is None:
	#	print("File incorrectly formetted")
	#	quit()
	#graph_problem, goal = graph_result

	graph_problem, goal = create_graph()

	# Extract parameter 2: "method" function used
	method = select_method(sys.argv[2])

	if method is None:
		print("Incorrect method type, valid methods:\nDFS, BFS, GBFS, AS")
		quit()

	result = method(graph_problem)

	# Output paramter 1
	print("filename=", sys.argv[1], sep="", end=" | ")
	# Output paramter 2
	print("method=", sys.argv[2], sep="")
	# \n
	# Ouput goal node
	print("goal=", goal, sep="", end= " | ")
	if (result is not None):
	# Output number (length of path)
		print("number of nodes=", len(result.solution()), sep="")
	# \n
	# Output path: list of nodes
		print("path=", result.solution(), sep="")
