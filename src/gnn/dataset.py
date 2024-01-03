import numpy as np
import sys
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from tqdm import trange

from ..graph.graphObject import GraphObject


def get_datasets(filename, batch_size=4, num_el=100, division=None):
    """
    It creates training, validation and testing DataLoader.
    :param filename: directory + starting name of the pde saved on the file
    :type filename: str
    :param batch_size: dimension of the batch
    :param num_el: number of element to read from file
    :param division: percentage of [training, validation, testing] division: the list can have 2 or 3 elements, for
                     instance: [80, 20]     --> 80% training and 20% validation
                               [80, 10, 10] --> 80% training, 10% validation and 10% testing
    :type division: list
    :return: tuple of torch_geometric DataLoaders
    """
    if division is None:
        division = [80, 20]
    if len(division) != 2 and len(division) != 3:
        raise Exception("List division must have length 2 or 3")
    if sum(division) != 100:
        raise Exception("Sum of division elements must be 100")

    end1 = int(division[0] / 100 * num_el)
    print("Reading training data...")
    training_data = GraphDataset(filename, 0, end1)
    print("Reading validation data...")
    test_dataloader = None
    if len(division) == 2:
        val_data = GraphDataset(filename, end1, num_el)
    else:
        end2 = int(division[1] / 100 * num_el)
        val_data = GraphDataset(filename, end1, end1 + end2)
        print("Reading testing data...")
        test_data = GraphDataset(filename, end1 + end2, num_el)
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    if len(division) == 2:
        return train_dataloader, val_dataloader
    return train_dataloader, val_dataloader, test_dataloader


class GraphDataset(Dataset):
    """
    Class that implements a torch_geometric Dataset.
    """

    def __init__(self, filename, start_idx, end_idx):
        """
        We assume that graph objects are saved as filename_idx.
        :param filename: directory + starting name of the graph saved on the file
        :type filename: str
        :param start_idx: index of starting graph to read
        :type start_idx: int
        :param end_idx: index of ending graph to read
        :type end_idx: int
        """
        super().__init__()
        self.objs = []
        for i in trange(end_idx - start_idx, ncols=100, file=sys.stdout):
            graph_obj = GraphObject.read(f"{filename}_{i + start_idx}")
            self.objs.append(get_torch_graph(vertexes=graph_obj.vertexes,
                                             edges=graph_obj.edges,
                                             in_features=graph_obj.in_features,
                                             out_features=graph_obj.out_features,
                                             edge_features=graph_obj.edge_features))

    def len(self):
        return len(self.objs)

    def get(self, idx):
        return self.objs[idx]


def get_torch_graph(vertexes, edges, in_features, out_features=None, edge_features=None):
    """
    It creates a pytorch_geometric Data instance, including node and edge features.
    :param vertexes: numpy array of shape (n_vertexes, n_dim)
    :type vertexes: np.ndarray
    :param edges: numpy array of shape (n_edges, 2)
    :type edges: np.ndarray
    :param in_features: numpy array of shape (n_vertexes, n_input) containing the input node features
    :type in_features: np.ndarray
    :param out_features: numpy array of shape (n_vertexes, n_output) containing the output node features
    :type out_features: np.ndarray
    :param edge_features: numpy array of shape (n_edges, n_edge_features) containing the edge features
    :type edge_features: np.ndarray
    :return: torch geometric Data object
    """
    empty = torch.empty(1)
    return Data(x=torch.tensor(in_features),  # Node features
                edge_index=torch.tensor(edges.T),  # Edges connectivity
                edge_attr=empty if edge_features is None else torch.tensor(edge_features),  # Edge features
                y=empty if out_features is None else torch.tensor(out_features),  # Node outputs
                pos=torch.tensor(vertexes))  # Node coordinates
