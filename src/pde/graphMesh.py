from dolfin import Mesh, MeshEditor
import numpy as np

from ..graph.graphND import GraphND
from ..graph.graph2D import Graph2D
from ..graph.graph3D import Graph3D


class GraphMesh:
    """
    Construct a dolfin mesh starting from a graph, using one-dimensional finite elements along the edges.
    """

    def __init__(self, graph, internal_nodes=0):
        """
        :param graph: GraphND object
        :type graph: GraphND
        :param internal_nodes: number of internal nodes for each edge
        """

        if not isinstance(graph, GraphND):
            raise Exception("Variable graph must be an instance of GraphND")

        self.graphProperties = graph.properties

        if self.graphProperties.n_dim != 2 and self.graphProperties.n_dim != 3:
            raise Exception("Only dimension 2 or 3 supported")

        # Refine mesh
        vertexes, edges = self.__refine(graph.vertexes, graph.edges, internal_nodes)

        # Create mesh
        self.__create_mesh(vertexes, edges)

    @staticmethod
    def __refine(vertexes, edges, internal_nodes):
        """
        Refine the graph adding internal nodes to create a mesh
        :param vertexes: vertexes of the graph
        :type vertexes: np.ndarray
        :param edges: edges of the graph
        :type edges: np.ndarray
        :param internal_nodes: number of internal nodes to add for each edge
        :type internal_nodes: int
        :return: refined vertexes and edges
        """
        new_vert = vertexes.tolist()
        new_edg = []
        edges_to_avoid = []
        for e in edges:
            i, j = e[0], e[1]
            # Self loops
            if i == j or [i, j] in edges_to_avoid:
                continue
            # Double connection
            elif any(np.equal(edges, [j, i]).all(1)):
                edges_to_avoid.append([j, i])
            # Normal case
            nodes = np.linspace(vertexes[i], vertexes[j], internal_nodes + 2)[1:-1]
            ii = i
            for n in nodes.tolist():
                # Insert node
                new_vert.append(n)
                jj = len(new_vert) - 1
                # Connect nodes
                new_edg.append([ii, jj])
                ii = jj
            new_edg.append([ii, j])

        return np.array(new_vert), np.array(new_edg)

    def __create_mesh(self, vertexes, edges):
        """
        Create dolfin mesh
        :param vertexes: vertexes of the mesh
        :type vertexes: np.ndarray
        :param edges: edges of the mesh
        :type edges: np.ndarray
        """
        editor = MeshEditor()
        self.mesh = Mesh()

        editor.open(self.mesh, type="interval", tdim=1, gdim=self.graphProperties.n_dim)

        # Add vertexes
        editor.init_vertices(vertexes.shape[0])
        for i, v in enumerate(vertexes):
            editor.add_vertex(i, v)

        # Add edges (cells)
        editor.init_cells(edges.shape[0])
        for i, e in enumerate(edges):
            editor.add_cell(i, e)

        editor.close()
        self.mesh.init()

    def __str__(self):
        return f"Vertexes are: \n {self.get_vertexes()}\n" \
               f"Edges are: \n {self.get_edges()}"

    def get_vertexes(self):
        """
        Get mesh vertexes
        :return: numpy array of vertexes
        """
        return np.array(self.mesh.coordinates())

    def get_edges(self):
        """
        Get mesh edges
        :return: numpy array of edges
        """
        return np.array(self.mesh.cells(), dtype=np.int64)

    def get_full_edges(self):
        """
        Get mesh edges ensuring that if (i,j) is present, also (j,i) will be present
        :return: numpy array of edges
        """
        edg = self.get_edges()
        add = np.zeros_like(edg)
        add[:, 0], add[:, 1] = edg[:, 1], edg[:, 0]
        return np.vstack((edg, add))

    def plot(self, *args, **kwargs):
        """
        Plot the mesh.
        """
        p = self.graphProperties
        v = self.get_vertexes()
        e = self.get_edges()
        p.n_vertexes = v.shape[0]
        p.directed = False
        if p.n_dim == 2:
            g = Graph2D(p, v, e)
        else:
            g = Graph3D(p, v, e)
        g.plot(*args, **kwargs)
