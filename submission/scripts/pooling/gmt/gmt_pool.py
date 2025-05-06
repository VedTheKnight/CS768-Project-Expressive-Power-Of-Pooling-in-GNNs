import torch
import torch.nn as nn
import torch.nn.functional as F

class GMTPooling(nn.Module):
    def __init__(self, in_channels, num_heads=4, out_channels=None, num_queries=1):
        super(GMTPooling, self).__init__()
        self.num_queries = num_queries
        self.out_channels = out_channels or in_channels

        # Learnable query embeddings
        self.queries = nn.Parameter(torch.randn(num_queries, self.out_channels))

        # Linear projection for keys and values
        self.key_proj = nn.Linear(in_channels, self.out_channels)
        self.value_proj = nn.Linear(in_channels, self.out_channels)

        # Transformer attention
        self.attn = nn.MultiheadAttention(self.out_channels, num_heads, batch_first=True)

    def forward(self, x, batch):
        """
        x: [num_nodes, in_channels]
        batch: [num_nodes]
        """
        # Separate graphs in the batch
        batch_size = batch.max().item() + 1
        outputs = []

        for i in range(batch_size):
            x_i = x[batch == i]  # [N_i, in_channels]
            k = self.key_proj(x_i).unsqueeze(0)    # [1, N_i, D]
            v = self.value_proj(x_i).unsqueeze(0)  # [1, N_i, D]
            q = self.queries.unsqueeze(0)          # [1, K, D]
            z, _ = self.attn(q, k, v)              # [1, K, D]
            outputs.append(z.squeeze(0))           # [K, D]

        z = torch.stack(outputs)  # [B, K, D]
        return z
