import numpy as np
from dolfin import UserExpression, Constant

from src.pde.ellipticPdeSolver2D import EllipticPdeSolver2D


class BoundaryExpression(UserExpression):

    def eval_cell(self, value, x, ufc_cell):
        if x[0] >= 0:
            value[0] = 2
        else:
            value[0] = 1

    def value_shape(self):
        return ()


class Problem(EllipticPdeSolver2D):
    def __init__(self, graph, internal_nodes=4):
        super().__init__(graph, internal_nodes)

    def solve(self, *args, **kwargs):
        # Define Dirichlet boundary conditions
        u0 = BoundaryExpression()

        def u0_boundary(x):
            tol = 1E-15
            for vertex in self.boundaryVertexes:
                if max(abs(vertex - x)) < tol:
                    return True
            return False

        # Define f and w
        f = Constant(0)
        w = Constant(1)
        # Solve the problem
        super().solve(w, f, u0, u0_boundary)


def get_node_features(n_vertexes, bound):
    """
    It computes input node features of the graph
    :param n_vertexes: number of vertexes of the graph
    :type n_vertexes: int
    :param bound: boolean numpy array, bound[i] is True if i-th vertex is on boundary
    :type bound: np.ndarray
    :return: numpy array of shape (n_vertexes, 2)
    """
    in_feat = np.zeros((n_vertexes, 2))
    in_feat[:, 0] = 1
    in_feat[0:bound.shape[0], 1] = bound[:, 0]
    return in_feat


def get_edge_features(vertexes, edges):
    """
    It computes edge features of the graph.
    :param vertexes: numpy array of shape (n_vertexes, n_dim)
    :type vertexes: np.ndarray
    :param edges: numpy array of shape (n_edges, 2)
    :type edges: np.ndarray
    :return: numpy array of shape (n_edges, n_dim + 1)
    """
    n_edges = edges.shape[0]
    n_dim = vertexes.shape[1]
    edge_attr = np.zeros((n_edges, n_dim + 1), dtype=np.float64)
    for i in range(n_edges):
        p1 = vertexes[edges[i, 0]]
        p2 = vertexes[edges[i, 1]]
        for j in range(n_dim):
            edge_attr[i, j] = (p1[j] + p2[j]) / 2
        edge_attr[i, n_dim] = np.linalg.norm(p1 - p2)
    return edge_attr
