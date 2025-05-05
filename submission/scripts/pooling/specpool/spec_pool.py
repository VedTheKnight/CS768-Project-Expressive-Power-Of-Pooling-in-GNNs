import torch
import torch.nn as nn
from torch_geometric.utils import get_laplacian, to_dense_adj, to_dense_batch, dense_to_sparse
from torch_geometric.nn import MessagePassing

class SpecPool(nn.Module):
    def __init__(self, in_channels, ratio=0.5, k_eigvecs=None):
        super(SpecPool, self).__init__()
        self.ratio = ratio
        self.k_eigvecs = k_eigvecs  # if None, inferred per graph
        self.in_channels = in_channels
        self.att = nn.Linear(in_channels, 1)

    def forward(self, x, edge_index, batch):
        # Dense adjacency and batch
        dense_x, mask = to_dense_batch(x, batch)
        dense_adj = to_dense_adj(edge_index, batch=batch)

        B, N, _ = dense_x.size()
        device = x.device

        pooled_x = []
        pooled_adj = []
        pooled_batch = []

        for i in range(B):
            xi = dense_x[i, mask[i]]
            Ai = dense_adj[i, :xi.size(0), :xi.size(0)]

            # Compute normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
            Di = torch.diag(Ai.sum(dim=-1))
            D_inv_sqrt = torch.pow(Di, -0.5)
            D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0
            L = torch.eye(xi.size(0), device=device) - D_inv_sqrt @ Ai @ D_inv_sqrt

            # Eigen decomposition
            eigvals, eigvecs = torch.linalg.eigh(L)
            k = min(int(self.ratio * xi.size(0)), xi.size(0))
            if self.k_eigvecs is not None:
                k = min(k, self.k_eigvecs)
            Uk = eigvecs[:, :k]  # spectral coordinates

            # Project node features to spectral space
            x_spec = Uk.T @ xi  # [k, F]

            pooled_x.append(x_spec)
            pooled_adj.append(torch.eye(k, device=device))  # assume fully connected in spectral space
            pooled_batch.append(torch.full((k,), i, device=device, dtype=torch.long))

        # Concatenate batches
        x_out = torch.cat(pooled_x, dim=0)  # [sum_k, F]
        adj_out, _ = dense_to_sparse(torch.block_diag(*pooled_adj))
        batch_out = torch.cat(pooled_batch, dim=0)

        return x_out, adj_out, batch_out
