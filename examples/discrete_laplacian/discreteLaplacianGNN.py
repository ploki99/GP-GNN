from torch import nn

from src.gnn.graphNeuralNetwork import GraphNeuralNetwork


class DiscreteLaplacianGNN(GraphNeuralNetwork):
    def __init__(self):
        in_node_features = 1
        edge_features = 0
        hidden_features = 1
        mp_steps = 1
        out_node_features = 1
        super().__init__(in_node_features, edge_features, hidden_features, mp_steps, out_node_features)

    def get_encoder_networks(self):
        # MLP for encoding edge features
        psi_e = nn.Sequential()
        # MLP for encoding node features
        psi_v = nn.Sequential()
        return psi_e, psi_v

    def get_mpb_networks(self):
        # MLP used for message step
        psi_e = nn.Sequential(
            nn.Linear(2 * self.hidden_features, self.hidden_features)
        )
        # MLP used for update step
        psi_v = nn.Sequential(
            nn.Linear(2 * self.hidden_features, self.hidden_features)
        )
        return psi_e, psi_v

    def get_decoder_network(self):
        # MLP for decoding node features
        psi = nn.Sequential()
        return psi
