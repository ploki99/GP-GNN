from abc import ABC, abstractmethod
from torch import nn

from .modules import MessagePassingBlock, Encoder, Decoder


class GraphNeuralNetwork(nn.Module, ABC):
    """
    Abstract Class for implementing our version of Graph Neural Network
    """

    def __init__(self, in_node_features, edge_features, hidden_features, mp_steps, out_node_features):
        """
        :param in_node_features: number of input node features
        :type in_node_features: int
        :param edge_features: number of edge features
        :type edge_features: int
        :param hidden_features: number of hidden features
        :type hidden_features: int
        :param mp_steps: number of message passing block steps
        :type mp_steps: int
        :param out_node_features: number of output node features
        :type out_node_features: int
        """
        super().__init__()
        self.in_node_features = in_node_features
        self.edge_features = edge_features
        self.hidden_features = hidden_features
        self.out_node_features = out_node_features
        # Encoder layer
        psi_e, psi_v = self.get_encoder_networks()
        self.encoder = Encoder(psi_e, psi_v)
        # Message passing layers
        self.mpb = nn.ModuleList()
        for i in range(mp_steps):
            psi_e, psi_v = self.get_mpb_networks()
            self.mpb.append(MessagePassingBlock(psi_e, psi_v, edge_features=edge_features > 0))
        # Decoder layer
        psi = self.get_decoder_network()
        self.decoder = Decoder(psi)

    def forward(self, vertexes, edge_index, edge_attr):
        """
        :param vertexes: torch tensor of shape (n_vertexes*batch_size, in_node_features)
        :type vertexes: torch.Tensor
        :param edge_index: torch tensor of shape (2, n_edges*batch_size)
        :type edge_index: torch.Tensor
        :param edge_attr: torch tensor of shape (n_edges*batch_size, edge_features)
        :type edge_attr: torch.Tensor
        :return: torch tensor of shape (n_vertexes*batch_size, out_node_features)
        :rtype: torch.Tensor
        """
        # Encoder
        v, e = self.encoder(vertexes, edge_attr)
        # Message passing block
        for m in self.mpb:
            v = m(v, edge_index, e)
        # Decoder
        return self.decoder(v)

    @abstractmethod
    def get_encoder_networks(self):
        """
        Define the psi_e and psi_v MLP used for the encoder.
        :return: tuple of pytorch nn.Sequential: [psi_e, psi_v]
        :rtype: tuple[nn.Sequential, nn.Sequential]
        """
        pass

    @abstractmethod
    def get_mpb_networks(self):
        """
        Define the psi_e and psi_v MLP used for the message passing layer.
        :return: tuple of pytorch nn.Sequential: [psi_e, psi_v]
        :rtype: tuple[nn.Sequential, nn.Sequential]
        """
        pass

    @abstractmethod
    def get_decoder_network(self):
        """
        Define the psi MLP used for the decoder.
        :return: instance of pytorch nn.Sequential
        :rtype: nn.Sequential
        """
        pass
