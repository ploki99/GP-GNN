from abc import ABC, abstractmethod
from dolfin import File
import numpy as np

from .graphMesh import GraphMesh
from ..graph.graphObject import GraphObject


class PdeSolver(ABC):
    """
    Abstract class to solve a generic pde problem defined on a graph.
    """
    def __init__(self, graph, internal_nodes=10):
        """
        :param graph: graph that represents the domain of the problem
        :param internal_nodes: number of internal nodes for each edge used to construct the mesh
        """
        self.solution = None
        self.mesh = GraphMesh(graph, internal_nodes=internal_nodes)
        self.boundaryVertexes = graph.get_boundary_coordinates()

    @abstractmethod
    def solve(self, *args, **kwargs):
        """
        Solve the pde problem.
        """
        pass

    def inf_error(self, exact_sol):
        """
        Compute error in infinite norm.
        :param exact_sol: python function that computes the exact solution
        :type exact_sol: function
        :return: error in infinite norm
        """
        if self.solution is None:
            raise Exception("You have to solve the problem first")

        diff_v = []
        for p in self.mesh.get_vertexes():
            diff_v.append(abs(exact_sol(p) - self.solution(p)))
        return float(max(diff_v))

    def plot_mesh(self, *args, **kwargs):
        """
        Plot the mesh used to solve the problem.
        """
        self.mesh.plot(*args, **kwargs)

    def get_solution(self):
        """
        Computes solution as list of numbers.
        :return: numpy array of solution at each vertex
        """
        if self.solution is None:
            raise Exception("You have to solve the problem first")

        u = []
        for p in self.mesh.get_vertexes():
            u.append([self.solution(p)])
        return np.array(u)

    def save_object(self, name, in_features, edge_features):
        """
        Save pde as object to a file.
        :param name: name of the file
        :type name: str
        :param in_features: input features
        :type in_features: np.ndarray
        :param edge_features: edge features
        :type edge_features: np.ndarray
        """
        obj = GraphObject(vertexes=self.mesh.get_vertexes(),
                          edges=self.mesh.get_full_edges(),
                          in_features=in_features,
                          out_features=self.get_solution(),
                          edge_features=edge_features)
        obj.save(name)

    def save_solution(self, name):
        """
        Save solution to a file.
        :param name: file name
        :type name: str
        """
        if self.solution is None:
            raise Exception("You have to solve the problem first")

        file = File(name + ".pvd")
        file << self.solution
