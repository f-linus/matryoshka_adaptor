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
n_epochs_supervised = 10

k = 20
learning_rate = 0.00001

query_batch_size = 800
n_non_matching_augmentations = 4
corpus_batch_size = 256
encoding_batch_size = 512

unsupervised_learning_model_file = "adaptor_hotpotqa_unsupervised.pt"
supervised_learning_model_file = "adaptor_hotpotqa_supervised.pt"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    corpus = datasets.load_dataset(dataset, "corpus", split="corpus")
    queries = datasets.load_dataset(dataset, "queries", split="queries")
    associations = datasets.load_dataset(dataset, split="train")

    original_model = SentenceTransformer(model_string)
    adaptor = MatryoshkaAdaptor(embedding_dim_full)

    # load model from adaptor.pt
    # adaptor.load_state_dict(torch.load("adaptor_nfcorpus_supervised.pt"))
    adaptor.to(device)

    training_loop(
        corpus=corpus,
        queries=queries,
        associations=associations,
        original_model=original_model,
        adaptor_model=adaptor,
        n_epochs_unsupervised=n_epochs_unsupervised,
        n_epochs_supervised=n_epochs_supervised,
        query_batch_size=query_batch_size,
        corpus_batch_size=corpus_batch_size,
        n_non_matching_augmentations=n_non_matching_augmentations,
        encoding_batch_size=encoding_batch_size,
        unsupervised_learning_model_file=unsupervised_learning_model_file,
        supervised_learning_model_file=supervised_learning_model_file,
        learning_rate_unsupervised=learning_rate,
        learning_rate_supervised=learning_rate / 20,
        gamma=10,
    )
