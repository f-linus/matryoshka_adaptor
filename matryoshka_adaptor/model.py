import torch
import torch.nn.functional as F
from torch import nn


class MatryoshkaAdaptor(nn.Module):
    def __init__(self, original_embedding_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.original_embedding_dim = original_embedding_dim

        # preliminary design
        self.layer1 = nn.Sequential(
            nn.Linear(original_embedding_dim, 2 * original_embedding_dim),
            nn.LayerNorm(2 * original_embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(2 * original_embedding_dim, 2 * original_embedding_dim),
            nn.LayerNorm(2 * original_embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.layer3 = nn.Sequential(
            nn.Linear(2 * original_embedding_dim, 2 * original_embedding_dim),
            nn.LayerNorm(2 * original_embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.layer4 = nn.Sequential(
            nn.Linear(2 * original_embedding_dim, 2 * original_embedding_dim),
            nn.LayerNorm(2 * original_embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.layer5 = nn.Linear(2 * original_embedding_dim, original_embedding_dim)

    def forward(self, original_embeddings: torch.Tensor):
        x = self.layer1(original_embeddings)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.layer5(x)


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
