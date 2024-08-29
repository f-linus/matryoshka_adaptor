import torch
import torch.nn.functional as F
from torch import nn


class MatryoshkaAdaptor(nn.Module):
    def __init__(self, original_embedding_dim: int):
        super().__init__()
        self.original_embedding_dim = original_embedding_dim

        # preliminary design
        self.linear1 = nn.Linear(original_embedding_dim, 2 * original_embedding_dim)
        self.linear2 = nn.Linear(
            2 * original_embedding_dim, int(0.5 * original_embedding_dim)
        )
        self.linear3 = nn.Linear(
            int(0.5 * original_embedding_dim), original_embedding_dim
        )

        self.relu = nn.ReLU()

    def forward(self, original_embeddings: torch.Tensor):
        delta = self.linear1(original_embeddings)
        delta = self.relu(delta)
        delta = self.linear2(delta)
        delta = self.relu(delta)
        delta = self.linear3(delta)
        return delta
