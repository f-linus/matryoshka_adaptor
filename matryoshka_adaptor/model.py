import torch
import torch.nn.functional as F
from torch import nn


class MatryoshkaAdaptor(nn.Module):
    def __init__(self, original_embedding_dim: int):
        super().__init__()
        self.original_embedding_dim = original_embedding_dim

        # preliminary design
        self.linear1 = nn.Linear(original_embedding_dim, 2 * original_embedding_dim)
        self.linear2 = nn.Linear(2 * original_embedding_dim, original_embedding_dim)

        self.relu = nn.ReLU()

    def forward(self, original_embeddings: torch.Tensor):
        delta = self.linear1(original_embeddings)
        delta = self.relu(delta)
        delta = self.linear2(delta)
        return delta


class SentenceTransformerTruncated:
    def __init__(self, model, embedding_length):
        self.model = model
        self.prompts = model.prompts  # unfortunately this is necessary
        self.embedding_length = embedding_length

    def encode(self, sentences, **kwargs):
        embeddings = self.model.encode(sentences, **kwargs)
        return embeddings[:, : self.embedding_length]


class AdaptedSentenceTransformer:
    def __init__(self, model, adaptor, embedding_length):
        self.model = model
        self.prompts = model.prompts
        self.adaptor = adaptor
        self.embedding_length = embedding_length

    def encode(self, sentences, **kwargs):
        embeddings_original = self.model.encode(sentences, **kwargs)
        embeddings_adapted = embeddings_original + self.adaptor.forward(
            embeddings_original
        )
        return embeddings_adapted[:, : self.embedding_length]
