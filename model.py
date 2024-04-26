import torch
from torch import nn
import torch.nn.functional as F

import torch_geometric
from tsl.nn.blocks.encoders import RNN
from tsl.nn.layers import NodeEmbedding, DiffConv, GATConv

class FGNN(nn.Module):
    def __init__(self, input_size = 1, n_nodes = 5, horizon = 1,
                 hidden_size: int = 16,
                 rnn_layers: int = 4,    #5 was good for tikhonov
                 gnn_kernel: int = 2):
        super().__init__()

        self.encoder = nn.Linear(input_size, hidden_size)

        self.node_embeddings = NodeEmbedding(n_nodes, hidden_size)

        self.time_nn = RNN(input_size=hidden_size,
                           hidden_size=hidden_size,
                           n_layers=rnn_layers,
                           cell='gru',
                           return_only_last_state=True)
        
        self.space_nn = GATConv(in_channels=hidden_size,
                                 out_channels=hidden_size)

        self.decoder = nn.LazyLinear(horizon)

    def forward(self, x, device):
        x = x.unsqueeze(3)
        # x: [batch time nodes features]
        batch = x.shape[0]
        x_enc = self.encoder(x)  # linear encoder: x_enc = xΘ + b
        x_emb = x_enc + self.node_embeddings()  # add node-identifier embeddings
        # pre1 = x_emb
        h = self.time_nn(x_emb)  # temporal processing: x=[b t n f] -> h=[b n f]
        # pre1 = h.reshape(-1, h.shape[-1])
        A = torch.ones(h.shape[1], h.shape[1])
        edge_index = (A > 0).nonzero().t().to(device)
        out = self.space_nn(h, edge_index = edge_index , need_weights = True)  # spatial processing
        z = out[0]
        pre2 = z
        alpha = out[1]
        alpha = torch_geometric.utils.to_dense_adj(edge_index = alpha[0], edge_attr = alpha[1]).squeeze()
        alpha = alpha.reshape(-1, batch).T

        pre1 = alpha
        
        x_out = self.decoder(z.flatten(start_dim = 1))  # linear decoder: z=[b n f] -> 
        return x_out, pre1, pre2