import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableClusterPool(nn.Module):
    def __init__(self, in_channels, num_clusters, hidden_dim=64):
        super(LearnableClusterPool, self).__init__()
        self.assign_net = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_clusters)
        )

    def forward(self, x, adj, mask=None):
        """
        x: Tensor of shape [B, N, F] (dense node features)
        adj: Tensor of shape [B, N, N] (dense adjacency matrix)
        mask: Optional mask of shape [B, N] (indicates valid nodes)
        """

        S = self.assign_net(x)  # (B, N, K)
        S = F.softmax(S, dim=-1)

        # Apply mask if provided
        if mask is not None:
            S = S * mask.unsqueeze(-1)

        # Cluster-wise feature aggregation
        x_pooled = torch.matmul(S.transpose(1, 2), x)  # (B, K, F)

        # Cluster-wise adjacency aggregation
        adj_pooled = torch.matmul(S.transpose(1, 2), torch.matmul(adj, S))  # (B, K, K)

        # Regularization loss on assignment
        aux_loss = self.regularization_loss(S, mask)

        return x_pooled, adj_pooled, aux_loss

    def regularization_loss(self, S, mask=None):
        """
        Entropy loss to encourage confident assignments and diversity loss
        to spread nodes across clusters.
        """
        eps = 1e-9
        # Entropy loss
        entropy = -S * torch.log(S + eps)
        if mask is not None:
            entropy = entropy * mask.unsqueeze(-1)
        entropy = entropy.sum() / (mask.sum() + eps)

        # Diversity loss (encourages orthogonality of S^T S)
        S_T_S = torch.bmm(S.transpose(1, 2), S)  # (B, K, K)
        I = torch.eye(S_T_S.size(-1)).to(S_T_S.device)
        diversity_loss = ((S_T_S / (mask.sum(dim=1, keepdim=True) + eps).unsqueeze(-1)) - I).pow(2).mean()

        return 0.1 * entropy + 0.1 * diversity_loss
