import logging

import datasets
import ptvsd
import torch
from sentence_transformers import SentenceTransformer

from matryoshka_adaptor.model import MatryoshkaAdaptor
from matryoshka_adaptor.train import training_loop

logger = logging.getLogger(__name__)


debug_mode = False

if debug_mode:
    print("Waiting for debugger to attach...")
    ptvsd.enable_attach(address=("localhost", 5678))
    ptvsd.wait_for_attach()

embedding_dim_full = 384
model_string = "sentence-transformers/all-MiniLM-L6-v2"

dataset = "mteb/hotpotqa"

n_epochs_unsupervised = 2
n_epochs_supervised = 5

k = 20
learning_rate = 0.0001

query_batch_size = 1024
n_non_matching_augmentations = 9
corpus_batch_size = 1024
encoding_batch_size = 512


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    corpus = datasets.load_dataset(dataset, "corpus", split="corpus")
    queries = datasets.load_dataset(dataset, "queries", split="queries")
    associations = datasets.load_dataset(dataset, split="train")

    original_model = SentenceTransformer(model_string)
    adaptor = MatryoshkaAdaptor(embedding_dim_full)
    adaptor.to(device)

    optimizer = torch.optim.Adam(adaptor.parameters(), lr=learning_rate)

    training_loop(
        corpus=corpus,
        queries=queries,
        associations=associations,
        original_model=original_model,
        adaptor_model=adaptor,
        optimizer=optimizer,
        n_epochs_unsupervised=n_epochs_unsupervised,
        n_epochs_supervised=n_epochs_supervised,
        query_batch_size=query_batch_size,
        corpus_batch_size=corpus_batch_size,
        n_non_matching_augmentations=n_non_matching_augmentations,
        encoding_batch_size=encoding_batch_size,
    )
