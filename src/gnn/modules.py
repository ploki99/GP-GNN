import numpy as np
import torch
from torch import nn
from torch_geometric.nn import MessagePassing


class Encoder(nn.Module):
    """
    Implement encoder module.
    """
    def __init__(self, psi_e, psi_v):
        """
        :param psi_e: MLP for encoding edge features
        :type psi_e: nn.Sequential
        :param psi_v: MLP for encoding node features
        :type psi_v: nn.Sequential
        """
        super().__init__()
        self.psi_e = psi_e
        self.psi_v = psi_v

    def forward(self, u, e):
        """
        :param u: input node feature tensor of shape (-1, node_features)
        :type u: torch.Tensor
        :param e: edge feature tensor of shape (-1, edge_features)
        :type e: torch.Tensor
        :return: tuple of encoded features with shapes: (-1, hidden_features) and (-1, hidden_features)
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        return self.psi_v(u), self.psi_e(e)


class MessagePassingBlock(MessagePassing):
    """
    Implement message passing layer.
    """
    def __init__(self, psi_e, psi_v, edge_features=True):
        """
        :param psi_e: MLP used for message step
        :type psi_e: nn.Sequential
        :param psi_v: MLP used for update step
        :type psi_v: nn.Sequential
        :param edge_features: if False do not use edge features
        """
        # Use "Add" aggregation
        super().__init__(aggr='add')
        self.psi_e = psi_e
        self.psi_v = psi_v
        self.edge_features = edge_features

    def message(self, x_i, x_j, edge_attr):
        """
        Implement message step
        :param x_i: tensor of node feature of node i
        :type x_i: torch.Tensor
        :param x_j: tensor of node feature of node j
        :type x_j: torch.Tensor
        :param edge_attr: tensor of edge features
        :type edge_attr: torch.Tensor
        :return: result of message step
        """
        if self.edge_features:
            return self.psi_e(torch.cat([x_i, x_j, edge_attr], dim=1))
        return self.psi_e(torch.cat([x_i, x_j], dim=1))

    def update(self, aggr_out, x):
        """
        Implement update step
        :param aggr_out: output of aggregation step
        :type aggr_out: torch.Tensor
        :param x: tensor of node feature
        :type x: torch.Tensor
        :return: result of update step
        """
        return self.psi_v(torch.cat([x, aggr_out], dim=1))

    def forward(self, x, edge_index, edge_attr):
        """
        :param x: input node feature tensor of shape (-1, hidden_features)
        :type x: torch.Tensor
        :param edge_index: tensor of shape (2, num_edges)
        :type edge_index: torch.Tensor
        :param edge_attr: edge features tensor of shape (-1, hidden_features)
        :type edge_attr: torch.Tensor
        :return: output node feature tensor of shape (-1, hidden_features)
        :rtype: torch.Tensor
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)


class Decoder(nn.Module):
    """
    Implement decoder module.
    """
    def __init__(self, psi):
        """
        :param psi: MLP for decoding node features
        :type psi: nn.Sequential
        """
        super().__init__()
        self.psi = psi

    def forward(self, x):
        """
        :param x: hidden node features tensor of shape (-1, hidden_features)
        :type x: torch.Tensor
        :return: decoded node features tensor of shape (-1, node_features)
        :rtype: torch.Tensor
        """
        return self.psi(x)


class EarlyStopping:
    """
    Class used to implement early stopping.
    """
    def __init__(self, patience=10, delta=0):
        """
        :param patience: number of epochs to wait before stop training
        :param delta: tolerance used to compare best_loss with current loss
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = np.Inf

    def __call__(self, val_loss):
        """
        Check whether training should stop.
        :param val_loss: validation loss at a certain epoch
        :type val_loss: float
        :return: True if training should stop
        """
        ret = False
        if self.best_loss - val_loss <= self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                ret = True
        else:
            self.counter = 0

        if val_loss < self.best_loss:
            self.best_loss = val_loss

        return ret
