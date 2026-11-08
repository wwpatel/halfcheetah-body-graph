import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Cheetah body graph. 6 nodes: 0=bthigh,1=bshin,2=bfoot,3=fthigh,4=fshin,5=ffoot
EDGE_INDEX = torch.tensor([
    [0, 1, 1, 2, 3, 4, 4, 5, 0, 3],
    [1, 0, 2, 1, 4, 3, 5, 4, 3, 0],
], dtype=torch.long)


class GraphFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, hidden_dim=32):
        # output feature size = 6 joints * hidden_dim
        super().__init__(observation_space, features_dim=6 * hidden_dim)
        self.hidden_dim = hidden_dim
        # stem+trunk: attention over the body graph
        self.gat1 = GATConv(2, hidden_dim)
        self.gat2 = GATConv(hidden_dim, hidden_dim)
        self.register_buffer("edge_index", EDGE_INDEX)

    def forward(self, obs):
        # obs is [batch, 17]. Pull the 6 joint angles (idx 2-7) and 6 joint velocities (idx 11-16).
        batch = obs.shape[0]
        angles = obs[:, 2:8]       # [batch, 6]
        vels = obs[:, 11:17]       # [batch, 6]
        # stack into node features: each joint = [angle, velocity]
        x = torch.stack([angles, vels], dim=2)   # [batch, 6, 2]
        # run the graph per sample in the batch
        outs = []
        for i in range(batch):
            h = self.gat1(x[i], self.edge_index)
            h = torch.relu(h)
            h = self.gat2(h, self.edge_index)
            outs.append(h.flatten())
        return torch.stack(outs)   # [batch, 6*hidden_dim]

