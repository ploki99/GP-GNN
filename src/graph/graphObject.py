import pickle
import numpy as np


class GraphObject:
    """
    Store a graph as object, including eventually the input/output node features and the edge features.
    """
    def __init__(self, vertexes, edges, in_features=None, out_features=None, edge_features=None):
        """
        :param vertexes: vertexes matrix of the graph
        :type vertexes: np.ndarray
        :param edges: edges matrix of the graph. For undirected graph ensure that both (i,j) and (j,i) are in edges
        :type edges: np.ndarray
        :param in_features: input node features
        :type in_features: np.ndarray
        :param out_features: output node features
        :type out_features: np.ndarray
        :param edge_features: edge features
        :type edge_features: np.ndarray
        """
        self.vertexes = vertexes
        self.edges = edges
        self.in_features = in_features
        self.out_features = out_features
        self.edge_features = edge_features

    def __str__(self):
        return f"Vertexes are {self.vertexes}\n" \
               f"Edges are {self.edges}\n" \
               f"Input features vector is {self.in_features}\n" \
               f"Output features vector is {self.out_features}\n" \
               f"Edge features vector is {self.edge_features}"

    def save(self, name):
        """
        Save object to file.
        :param name: name of the file
        :type name: str
        """
        with open(name + '.pkl', 'wb') as out:
            pickle.dump(self, out, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def read(cls, name):
        """
        Read object from file.
        :param name: name of the file
        :type name: str
        :return: GraphObject read from file
        """
        with open(name + '.pkl', 'rb') as inp:
            obj = pickle.load(inp)
            if not isinstance(obj, cls):
                raise Exception("Only GraphObject type allowed")
            return obj

