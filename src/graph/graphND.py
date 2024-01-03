import numpy as np
import random

from .graphProperties import GraphProperties


class GraphND:
    """
    Class for generating random graph given an instance of GraphProperties.
    """

    def __init__(self, properties=None, vertexes=None, edges=None):
        """
        :param properties: instance of class GraphProperties
        :type properties: GraphProperties
        :param vertexes: if not None use these vertexes and do not generate others
        :type vertexes: list | np.ndarray
        :param edges: if not None and vertexes are passed, use these edges and do not generate others
        :type edges: list | np.ndarray
        """
        # Check minimum requirements
        if properties is None and (vertexes is None or edges is None):
            raise Exception("Not enough information to generate the graph")
        # Check properties
        if properties is None:
            properties = GraphProperties.get_properties_from_graph(vertexes, edges)
        if not isinstance(properties, GraphProperties):
            raise Exception("Variable properties must be an instance of GraphProperties")
        self.properties = properties
        # Check vertexes
        if vertexes is None:
            self.__generate_vertexes()
        else:
            self.vertexes = np.array(vertexes)
            # Check vertexes correctness
            if self.vertexes.shape != (self.properties.n_vertexes, self.properties.n_dim):
                raise Exception("Vertexes matrix must have shape (properties.n_vertexes, properties.n_dim)")
        # Check edges
        if edges is None:
            self.__generate_edges()
        else:
            edges = np.array(edges)
            # Check edges correctness
            if edges.shape[1] != 2 or not issubclass(edges.dtype.type, np.integer):
                raise Exception("Edges must be a list of item, each element in the form [n,m] with n and m integer")
            if np.min(edges) < 0 or np.max(edges) > self.properties.n_vertexes - 1:
                raise Exception("Edges list has a vertex not present in vertexes matrix")
            # Save given edges
            self.__save_edges(edges)

    def __str__(self):
        return f"Vertexes of the graph are:\n {self.vertexes}\n" \
               f"Edges of the graph are:\n {self.edges}\n" \
               f"{self.properties.__str__()}"

    # Private interface of the class

    def __save_edges(self, edges):
        """
        Save given edges of the graph: if not directed only one edge per connection is saved
        :param edges: edges of the graph
        :type edges: np.ndarray
        """
        if self.properties.directed:
            self.edges = edges
        else:
            single_edges = []
            for e in edges:
                if not ([e[0], e[1]] in single_edges or [e[1], e[0]] in single_edges):
                    single_edges.append([e[0], e[1]])
            self.edges = np.array(single_edges)

    def __generate_vertexes(self):
        """
        Generate random vertex coordinates given properties specified before.
        """
        self.vertexes = np.zeros((self.properties.n_vertexes, self.properties.n_dim))
        for k in range(self.properties.n_vertexes):
            for i in range(self.properties.n_dim):
                self.vertexes[k][i] = random.uniform(self.properties.limits[i][0],
                                                     self.properties.limits[i][1])

    def __generate_edges(self):
        """
        Generate random edges given the properties specified before.
        """
        # Maximum number of edges addable
        max_edges = self.properties.get_max_edges()
        # Number of edges to add that avoids cycles
        acyclic_edges = 0
        if self.properties.weak_connected:
            acyclic_edges = self.properties.n_vertexes - 1
            if self.properties.strong_connected:
                acyclic_edges += 1
        elif self.properties.acyclic:
            acyclic_edges = min(self.properties.n_vertexes - 1, self.properties.minimum_edges)
        # Number of random edges to add
        random_edges = max(min(max_edges, self.properties.minimum_edges) - acyclic_edges, 0)
        # Initialize and fill edges matrix
        self.edges = np.zeros(shape=(acyclic_edges + random_edges, 2), dtype=np.int64)
        self.__add_acyclic_edges(acyclic_edges, self.properties.strong_connected)
        for k in range(random_edges):
            self.__add_random_edge(acyclic_edges + k, self.properties.directed and self.properties.acyclic)

    def __add_acyclic_edges(self, n_edges, strong_connect=False):
        """
        Add n_edges to the graph in such a way we don't have cycles.
        Note that the n_edges must be the first edges added in the graph.
        :param n_edges: number of edges to add
        :type n_edges: int
        :param strong_connect: if True then it creates a strong connected graph with one cycle
        """
        # In case of strong connected graph, we add the last edges after the loop
        if strong_connect:
            n_edges -= 1
        unselected_v = [i for i in range(self.properties.n_vertexes)]
        i = random.randrange(0, len(unselected_v))
        selected_v = [unselected_v[i]]
        unselected_v.pop(i)
        for k in range(n_edges):
            i = random.randrange(0, len(unselected_v))
            if strong_connect:
                # Take the index of the last element connected to the graph, since we will connect the new vertex to it
                j = len(selected_v) - 1
            else:
                j = random.randrange(0, len(selected_v))
            # Add edge
            self.edges[k] = [unselected_v[i], selected_v[j]]
            selected_v.append(unselected_v[i])
            unselected_v.pop(i)
        # Add last edge
        if strong_connect:
            self.edges[n_edges] = [selected_v[0], selected_v[-1]]

    def __add_random_edge(self, idx, keep_acyclicity=False):
        """
        Add a random edge to the graph.
        :param idx: index of the row of the edge matrix where to add the edge
        :type idx: int
        :param keep_acyclicity: if True add an edge keeping the graph acyclic
        """
        while True:
            a = random.randrange(0, self.properties.n_vertexes)
            while True:
                b = random.randrange(0, self.properties.n_vertexes)
                if self.properties.directed and self.properties.loop:
                    break
                if a != b:
                    break
            if not self.__is_edge_present([a, b]):
                if keep_acyclicity and self.__is_edge_present([b, a]):
                    continue
                self.edges[idx] = [a, b]
                if keep_acyclicity and self.__has_dgraph_cycle():
                    self.edges[idx] = [b, a]
                break

    def __is_edge_present(self, e):
        """
        Verify if an edge is present in the graph.
        :param e: edge to verify in the form [a,b]
        :type e: list
        :return: True if the edge is present
        """
        if self.properties.directed:
            return any(np.equal(self.edges, e).all(1))
        return any(np.equal(self.edges, e).all(1)) or any(np.equal(self.edges, [e[1], e[0]]).all(1))

    def __has_dgraph_cycle(self):
        """
        Implement a topological sort to verify if the graph has a cycle.
        :return: True if directed graph has at least a cycle
        """
        # Create a copy of list self.edges
        graph = self.edges.tolist()

        def has_no_incoming_edge(node):
            for e in graph:
                if e[1] == node:
                    return False
            return True

        # Initialize empty list L
        L = []
        # Create set of nodes with no incoming edges
        Q = []
        for i in range(self.properties.n_vertexes):
            if has_no_incoming_edge(i):
                Q.append(i)
        # While Q is not empty
        while len(Q) > 0:
            n = Q.pop(0)
            L.append(n)
            # For each node m with an edge e from n to m
            for m in range(self.properties.n_vertexes):
                if [n, m] in graph:
                    # Remove edge e from the graph
                    graph.remove([n, m])
                    # If m has no other incoming edges
                    if has_no_incoming_edge(m):
                        Q.append(m)
        if len(graph) > 0:
            return True
        # Proposed topologically sorted order: L
        return False

    # Public interface of the class

    def get_edges(self):
        """
        Return a copy of the edge matrix, in case of undirected graph, it duplicates the edges so if (i,j) is present,
        also (j,i) will be in the matrix
        :return: numpy array of edges
        """
        if self.properties.directed:
            return np.array(self.edges.copy())
        add = np.zeros_like(self.edges)
        add[:, 0], add[:, 1] = self.edges[:, 1], self.edges[:, 0]
        return np.vstack([self.edges, add])

    def get_n_cycles(self):
        """
        Compute and return the number of cycles in the graph
        :return: number of cycles
        """
        if self.properties.n_cycles == -1:
            self.properties.update_n_cycles(self.vertexes, self.edges)
        return self.properties.n_cycles

    def get_graph_laplacian(self):
        """
        Compute graph laplacian.
        :return: numpy array containing the graph laplacian
        """
        L = np.zeros((self.properties.n_vertexes, self.properties.n_vertexes), dtype=np.int64)
        for e in self.edges:
            L[e[0], e[0]] += 1
            L[e[0], e[1]] = -1
        if not self.properties.directed:
            for e in self.edges:
                L[e[1], e[1]] += 1
                L[e[1], e[0]] = -1
        return L

    def get_discrete_laplacian(self, x):
        """
        Compute discrete laplacian of vector x.
        :param x: numpy array with shape (n_vertexes, 1)
        :type x: np.ndarray
        :return: numpy array with shape (n_vertexes, 1)
        """
        res = np.zeros_like(x)
        A = []
        for i in range(self.properties.n_vertexes):
            A.append([])
        for e in self.edges:
            A[e[0]].append(e[1])
            A[e[1]].append(e[0])
        for i in range(self.properties.n_vertexes):
            for j in A[i]:
                res[i] += x[i] - x[j]
        return res

    def get_boundary_mask(self):
        """
        Compute naturally boundary vertexes:
            undirected graph --> nodes with at maximum one connection
            directed graph --> nodes with no incoming edges, not counting the self loops
        :return: boolean numpy array
        """
        n_connections = np.zeros((self.properties.n_vertexes, 1), dtype=int)
        for e in self.edges:
            if e[0] != e[1]:
                if not self.properties.directed:
                    n_connections[e[0]] += 1
                n_connections[e[1]] += 1
        mask = np.array(n_connections <= 1 - int(self.properties.directed), dtype=bool)
        return mask

    def get_boundary_coordinates(self):
        """
        Compute naturally boundary vertexes coordinates:
            undirected graph --> nodes with at maximum one connection
            directed graph --> nodes with no incoming edges, not counting the self loops
        :return: numpy array of coordinates
        """
        mask = self.get_boundary_mask()
        coord = np.zeros((np.sum(mask), self.properties.n_dim))
        j = 0
        for i, v in enumerate(self.vertexes):
            if mask[i]:
                coord[j] = v
                j += 1
        return coord

    @classmethod
    def get_default_graph(cls, n_vertexes):
        """
        Generate a random graph with default options, given the number of vertexes.
        :param n_vertexes: number of vertexes in the graph
        :type n_vertexes: int
        :return: GraphND instance
        """
        p = GraphProperties(n_vertexes)
        return cls(p)
