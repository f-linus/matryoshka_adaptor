import logging

import datasets
import matplotlib.pyplot as plt
import ptvsd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch import nn

from matryoshka_adaptor.model import MatryoshkaAdaptor

logger = logging.getLogger(__name__)

debug_mode = False

if debug_mode:
    print("Waiting for debugger to attach...")
    ptvsd.enable_attach(address=("localhost", 5678))
    ptvsd.wait_for_attach()


def batch_iterator(
    training_corpus, training_dataset, batch_size, device, original_model
):
    n_batches = len(training_dataset) // batch_size
    for i in range(0, len(training_dataset), batch_size):
        batch = training_corpus[training_dataset[i : i + batch_size]["corpus-id"]][
            "text"
        ]
        embeddings_np = original_model.encode(batch)
        embeddings = torch.Tensor(embeddings_np).to(device)
        del embeddings_np
        logger.info(f"Batch {int(i/batch_size)}/{n_batches}")
        yield embeddings


def loss_fn(
    embeddings_original: torch.Tensor,
    embeddings_adapted: torch.Tensor,
    k=10,
    alpha=1.0,
    beta=1.0,
    gamma=1.0,
):
    device = embeddings_original.device

    # first term: pair-wise similarity differences for different truncation lengths
    normalized_embeddings_original = F.normalize(embeddings_original, dim=1, p=2.0)
    original_similarties = torch.matmul(
        normalized_embeddings_original, normalized_embeddings_original.t()
    )

    # a 3d tensor repeating the embedding matrix as many times as we have different truncations
    adapted_truncations = embeddings_adapted.repeat(embeddings_adapted.shape[1], 1, 1)

    # create an index among the individual embeddings
    index = torch.arange(adapted_truncations.shape[2]).unsqueeze(0).unsqueeze(0)
    index = index.expand(adapted_truncations.shape[0], adapted_truncations.shape[1], -1)

    # create an index among the different truncation levels
    row_indices = torch.arange(adapted_truncations.shape[0]).unsqueeze(1).unsqueeze(2)

    # we now mask of the 3d tensor of copies to have different truncation lengths
    mask = index < (adapted_truncations.shape[2] - row_indices)

    # move to device
    mask = mask.to(device)
    adapted_truncations = torch.where(
        mask, adapted_truncations, torch.zeros_like(adapted_truncations)
    )

    adapted_truncations_normalised = F.normalize(adapted_truncations, dim=2, p=2.0)

    # cosine similarities at different truncation lengths
    adapted_truncation_similarities = torch.bmm(
        adapted_truncations_normalised, adapted_truncations_normalised.transpose(1, 2)
    )
    deltas = (adapted_truncation_similarities - original_similarties.unsqueeze(0)).abs()
    first_loss_term = deltas.sum()

    # second term: pair-wise similarity differences but only for the neighbourhood of original embeddings

    # for each original embedding, we create a mask two only consider the k closest embeddings
    # with that mask we can ignore all similarity differences for embeddings that are not respective neighbourhoods
    topk_indices = original_similarties.topk(k, dim=1).indices
    mask = torch.zeros_like(original_similarties)
    mask = mask.scatter_(1, topk_indices, 1)

    # masked similarity differences
    masked_deltas = mask.unsqueeze(0) * deltas
    second_loss_term = masked_deltas.sum()

    # third term: reconstruction loss, i.e. difference between original and adapted embeddings
    third_loss_term = (embeddings_original - embeddings_adapted).abs().sum()

    return alpha * first_loss_term + beta * second_loss_term + gamma * third_loss_term


def training_loop(
    training_corpus,
    training_dataset,
    original_model,
    adaptor,
    device,
    optimizer,
    batch_size,
    epochs,
):
    losses = []

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch}/{epochs}")
        for embeddings in batch_iterator(
            training_corpus, training_dataset, batch_size, device, original_model
        ):
            adaption_delta = adaptor.forward(embeddings)
            embeddings_new = embeddings + adaption_delta
            loss = loss_fn(embeddings, embeddings_new, k=k)
            losses.append(loss.item())
            logger.info(f"Loss: {loss}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return losses


embedding_dim_full = 384
model_string = "sentence-transformers/all-MiniLM-L6-v2"
batch_size = 512
learning_rate = 0.001
k = 10
epochs = 2

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    training_corpus = datasets.load_dataset("mteb/msmarco", "corpus", split="corpus")
    training_dataset = datasets.load_dataset("mteb/msmarco", split="train")
    original_model = SentenceTransformer(model_string)

    adaptor = MatryoshkaAdaptor(embedding_dim_full)

    # move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adaptor = adaptor.to(device)
    original_model = original_model.to(device)

    # training
    optimizer = torch.optim.Adam(adaptor.parameters(), lr=learning_rate)
    losses = training_loop(
        training_corpus,
        training_dataset,
        original_model,
        adaptor,
        device,
        optimizer,
        batch_size,
        epochs,
    )

    # save model
    torch.save(adaptor.state_dict(), "adaptor.pt")

    # save loss curve
    plt.plot(losses, color="blue")
    plt.savefig("loss.png", dpi=300)
