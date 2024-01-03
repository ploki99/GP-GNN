from torch import nn

from src.gnn.graphNeuralNetwork import GraphNeuralNetwork


class PdeGNN(GraphNeuralNetwork):
    def __init__(self):
        in_node_features = 2
        edge_features = 3
        hidden_features = 5
        mp_steps = 40
        out_node_features = 1
        super().__init__(in_node_features, edge_features, hidden_features, mp_steps, out_node_features)

    def get_encoder_networks(self):
        # MLP for encoding edge features
        psi_e = nn.Sequential(
            nn.Linear(self.edge_features, self.hidden_features)
        )
        # MLP for encoding node features
        psi_v = nn.Sequential(
            nn.Linear(self.in_node_features, self.hidden_features)
        )
        return psi_e, psi_v

    def get_mpb_networks(self):
        # MLP used for message step
        psi_e = nn.Sequential(
            nn.Linear(3 * self.hidden_features, 2 * self.hidden_features),
            nn.ReLU(),
            nn.Linear(2 * self.hidden_features, self.hidden_features)
        )
        # MLP used for update step
        psi_v = nn.Sequential(
            nn.Linear(2 * self.hidden_features, 4 * self.hidden_features),
            nn.ReLU(),
            nn.Linear(4 * self.hidden_features, self.hidden_features)
        )
        return psi_e, psi_v

    def get_decoder_network(self):
        # MLP for decoding node features
        psi = nn.Sequential(
            nn.Linear(self.hidden_features, self.out_node_features)
        )
        return psi
