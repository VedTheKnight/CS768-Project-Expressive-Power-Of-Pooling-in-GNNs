import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleSoftPool(nn.Module):
    def __init__(self, in_channels, num_clusters):
        super(SimpleSoftPool, self).__init__()
        self.assign_mlp = nn.Linear(in_channels, num_clusters)
        self.num_clusters = num_clusters

    def forward(self, x, adj, mask=None):
        """
        x: [B, N, F]       (dense padded features)
        adj: [B, N, N]     (dense adjacency)
        mask: [B, N]       (optional mask for valid nodes)
        """
        # Learn soft assignments
        S = F.softmax(self.assign_mlp(x), dim=-1)  # [B, N, K]

        # Pooled node features: X' = S.TX
        x_pool = torch.matmul(S.transpose(1, 2), x)  # [B, K, F]

        # Pooled adjacency: A' = S.TAS
        adj_pool = torch.matmul(S.transpose(1, 2), torch.matmul(adj, S))  # [B, K, K]

        entropy = (-S * torch.log(S + 1e-10)).sum(dim=-1)  # [B, N]
        if mask is not None:
            entropy = (entropy * mask).sum() / mask.sum()
        else:
            entropy = entropy.mean()

        return x_pool, adj_pool, entropy
