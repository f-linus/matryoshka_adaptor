import logging

import datasets
import matplotlib.pyplot as plt
import pandas as pd
import ptvsd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch import nn

from matryoshka_adaptor.model import MatryoshkaAdaptor

logger = logging.getLogger(__name__)


def batch_iterator(
    corpus,
    queries,
    associations,
    batch_size,
    device,
    original_model,
    encoding_batch_size=512,
):
    corpus_ids_to_select = set(associations[:]["corpus-id"])
    query_ids_to_select = set(associations[:]["query-id"])
    size_ratio = len(corpus_ids_to_select) / len(query_ids_to_select)

    corpus = corpus.filter(lambda x: x["_id"] in corpus_ids_to_select)
    queries = queries.filter(lambda x: x["_id"] in query_ids_to_select)

    # shuffle corpus and queries
    corpus = corpus.shuffle(seed=0)
    queries = queries.shuffle(seed=0)

    # braindead way of balancing query and corpus instances in the batch
    corpus_batch_portion = size_ratio * batch_size
    queries_batch_portion = (1 / size_ratio) * batch_size
    factor = batch_size / (queries_batch_portion + corpus_batch_portion)
    corpus_batch_portion = int(corpus_batch_portion * factor)
    queries_batch_portion = batch_size - corpus_batch_portion

    for corpus_ptr, queries_ptr in zip(
        range(0, len(corpus), corpus_batch_portion),
        range(0, len(queries), queries_batch_portion),
    ):
        corpus_batch = corpus[corpus_ptr : corpus_ptr + corpus_batch_portion]["text"]
        queries_batch = queries[queries_ptr : queries_ptr + queries_batch_portion][
            "text"
        ]
        batch = corpus_batch + queries_batch

        # ditch batch if smaller than batch size (for consistent loss func.)
        if len(batch) < batch_size:
            continue

        embeddings_np = original_model.encode(
            batch, batch_size=encoding_batch_size, show_progress_bar=False
        )
        embeddings = torch.Tensor(embeddings_np).to(device)
        del embeddings_np
        yield embeddings


def batch_iterator_corpus_only(
    corpus,
    queries,
    associations,
    batch_size,
    device,
    original_model,
    encoding_batch_size=512,
):
    corpus = corpus.shuffle(seed=0)
    for corpus_ptr in range(0, len(corpus), batch_size):
        corpus_batch = corpus[corpus_ptr : corpus_ptr + batch_size]["text"]

        # ditch batch if smaller than batch size (for consistent loss func.)
        if len(corpus_batch) < batch_size:
            continue

        embeddings_np = original_model.encode(
            corpus_batch, batch_size=encoding_batch_size, show_progress_bar=False
        )
        embeddings = torch.Tensor(embeddings_np).to(device)
        del embeddings_np
        yield embeddings


def loss_unsupervised(
    embeddings_original: torch.Tensor,
    embeddings_adapted: torch.Tensor,
    k=10,
    alpha=1.0,
    beta=1.0,
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
    index = (
        torch.arange(adapted_truncations.shape[2], device=device)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    index = index.expand(adapted_truncations.shape[0], adapted_truncations.shape[1], -1)

    # create an index among the different truncation levels
    row_indices = (
        torch.arange(adapted_truncations.shape[0], device=device)
        .unsqueeze(1)
        .unsqueeze(2)
    )

    # we now mask of the 3d tensor of copies to have different truncation lengths
    mask = index < (adapted_truncations.shape[2] - row_indices)

    # move to device
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

    return first_loss_term + alpha * second_loss_term + beta * third_loss_term


def training_loop_unsupervised(
    corpus,
    queries,
    associations,
    original_model,
    adaptor,
    device,
    optimizer,
    batch_size,
    epochs,
    log_interval=10,
):
    losses = []

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        for step, embeddings in enumerate(
            batch_iterator(
                corpus,
                queries,
                associations,
                batch_size,
                device,
                original_model,
            )
        ):
            adaption_delta = adaptor.forward(embeddings)
            embeddings_new = embeddings + adaption_delta
            loss = loss_unsupervised(embeddings, embeddings_new)
            losses.append(loss.item())
            if step % log_interval == 0:
                logger.info(f"Step {step}, Loss: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return losses


def training_loop_supervised(
    corpus,
    queries,
    associations,
    original_model,
    adaptor,
    device,
    optimizer,
    batch_size,
    epochs,
    log_interval=10,
):
    losses = []

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")

        # iterate as long as both batch iterators yield a batch
        for embeddings_unsupervised, embeddings_supervised in zip(
            batch_iterator(
                corpus,
                queries,
                associations,
                batch_size,
                device,
                original_model,
            ),
            supervised_batch_iterator(
                corpus,
                queries,
                associations,
                batch_size,
                device,
                original_model,
            ),
        ):
            adaption_delta = adaptor.forward(embeddings_unsupervised)
            embeddings_new = embeddings_unsupervised + adaption_delta

            query_e

            loss = loss_unsupervised(
                embeddings_unsupervised, embeddings_new
            ) + loss_supervised(embeddings_supervised)
            losses.append(loss.item())
            if step % log_interval == 0:
                logger.info(f"Step {step}, Loss: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return losses


def supervised_batch_iterator(
    corpus,
    queries,
    associations,
    query_batch_size,
    device,
    model,
    n_non_matching_augmentations=9,
):
    associations_df = associations.to_pandas()
    corpus_df = corpus.to_pandas()
    queries_df = queries.to_pandas()

    corpus_df.set_index("_id", inplace=True)
    queries_df.set_index("_id", inplace=True)

    associations_df = associations_df.join(corpus_df, on="corpus-id", how="left")
    associations_df = associations_df.join(
        queries_df, on="query-id", how="left", rsuffix="_query"
    )

    for batch_ptr in range(0, len(associations_df), query_batch_size):
        batch = associations_df.iloc[batch_ptr : batch_ptr + query_batch_size]

        # we add non matching documents to the queries for somewhat contrastive training
        # in theory, documents added with score 0 could actually be part of the matching ones
        # however, for sufficiently large corpi prob. is -> 0 and we prefer to not do the extra computation expenditure
        zero_score_augmentations = pd.DataFrame(
            {
                "text_query": batch["text_query"].to_list()
                * n_non_matching_augmentations,
                "text": associations_df["text"]
                .sample(n=len(batch) * n_non_matching_augmentations)
                .to_list(),
            }
        )
        zero_score_augmentations["score"] = 0

        batch_augmented = pd.concat([batch, zero_score_augmentations])

        query_embeddings_np = model.encode(batch["text_query"].to_list())
        document_embeddings_np = model.encode(batch_augmented["text"].to_list())
        scores = batch_augmented["score"].to_list()

        # move to device
        query_embeddings = torch.Tensor(query_embeddings_np).to(device)
        document_embeddings = torch.Tensor(document_embeddings_np).to(device)
        scores = torch.Tensor(scores).to(device)

        # reshaping and re-ordering
        document_embeddings = document_embeddings.reshape(
            1 + n_non_matching_augmentations, query_batch_size, -1
        )
        document_embeddings = document_embeddings.transpose(0, 1)

        scores = scores.reshape(1 + n_non_matching_augmentations, query_batch_size)
        scores = scores.transpose(0, 1)

        yield (
            query_embeddings,
            document_embeddings,
            scores,
        )


def loss_supervised(
    query_embeddings_adapted, document_embeddings_adapted, scores_batched
):
    device = query_embeddings_adapted.device

    # rank loss
    batch_size = query_embeddings_adapted.shape[0]

    max_embedding_dim = query_embeddings_adapted.shape[1]

    loss_sum = 0
    for i in range(batch_size):
        query_embedding = query_embeddings_adapted[i]
        document_embeddings = document_embeddings_adapted[i]
        scores = scores_batched[i]

        # this represents the indicator funciton in the loss term
        relevancy_mask = scores.unsqueeze(0) > scores.unsqueeze(1)

        # this represents the difference between the scores, that we use to scale the rank loss
        # for MSMarco this will be irrelavent since equal to the relevancy mask (due to scores being 0 or 1)
        diff_matrix = scores.unsqueeze(0) - scores.unsqueeze(1)

        # here we need to compute the cosine similarities at different truncation levels
        flipped_triangular = torch.flip(
            torch.tril(torch.ones(max_embedding_dim, max_embedding_dim, device=device)),
            dims=(0,),
        )

        # expand query embeddings for different truncation levels
        query_embedding_truncation_matrix = query_embedding.expand(
            max_embedding_dim, -1
        )
        query_embedding_truncation_matrix = (
            query_embedding_truncation_matrix * flipped_triangular
        )

        # expand document embeddings for different truncation levels
        mask = flipped_triangular.repeat_interleave(
            document_embeddings.shape[0], dim=0
        ).reshape(max_embedding_dim, -1, max_embedding_dim)

        document_embeddings_truncation_tensor = (
            document_embeddings.expand(
                max_embedding_dim, document_embeddings.shape[0], -1
            )
            * mask
        )

        # normalise
        query_embedding_truncation_matrix_norm = F.normalize(
            query_embedding_truncation_matrix, dim=1, p=2.0
        )
        document_embeddings_truncation_tensor_norm = F.normalize(
            document_embeddings_truncation_tensor, dim=2, p=2.0
        )

        query_embedding_truncation_matrix_norm = (
            query_embedding_truncation_matrix_norm.reshape(
                max_embedding_dim, 1, max_embedding_dim
            )
        )
        cosine_similarities = torch.bmm(
            query_embedding_truncation_matrix_norm,
            document_embeddings_truncation_tensor_norm.transpose(1, 2),
        ).reshape(max_embedding_dim, document_embeddings.shape[0])

        # TODO: check if the order is correct here
        pairwise_cosim_differences = cosine_similarities.unsqueeze(
            2
        ) - cosine_similarities.unsqueeze(1)

        # log(1 + exp(x))
        pairwise_cosim_differences = torch.log(
            1 + torch.exp(pairwise_cosim_differences)
        )

        # expand the relevancy andd difference matrix to all truncation levels
        relevancy_mask_expanded = relevancy_mask.unsqueeze(0).expand_as(
            pairwise_cosim_differences
        )
        diff_matrix_expanded = diff_matrix.unsqueeze(0).expand_as(
            pairwise_cosim_differences
        )

        # aggregate over truncation levels
        loss_sum += (
            relevancy_mask_expanded * diff_matrix_expanded * pairwise_cosim_differences
        ).sum()

    return loss_sum


debug_mode = True

if debug_mode:
    print("Waiting for debugger to attach...")
    ptvsd.enable_attach(address=("localhost", 5678))
    ptvsd.wait_for_attach()

embedding_dim_full = 384
model_string = "sentence-transformers/all-MiniLM-L6-v2"
batch_size = 1024
learning_rate = 0.0001
k = 20
epochs = 12
n_non_matching_augmentations = 9
adaptor_file_to_load = "adaptor.pt"
adaptor_file_to_save = "adaptor_supervised.pt"
dataset = "mteb/hotpotqa"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    corpus = datasets.load_dataset(dataset, "corpus", split="corpus")
    queries = datasets.load_dataset(dataset, "queries", split="queries")
    associations = datasets.load_dataset(dataset, split="train")

    original_model = SentenceTransformer(model_string)

    adaptor = MatryoshkaAdaptor(embedding_dim_full)

    # load adaptor
    adaptor.load_state_dict(torch.load(adaptor_file_to_load))

    adaptor = adaptor.to(device)
    original_model = original_model.to(device)

    # get one supervised batch (for testing)
    query_embeddings, document_embeddings, scores = next(
        supervised_batch_iterator(
            corpus,
            queries,
            associations,
            batch_size,
            device,
            original_model,
            n_non_matching_augmentations,
        )
    )

    query_embeddings_adapted = adaptor.forward(query_embeddings)
    document_embeddings_adapted = adaptor.forward(
        document_embeddings.reshape(-1, embedding_dim_full)
    )
    document_embeddings_adapted = document_embeddings_adapted.reshape(
        query_embeddings.shape[0], 1 + n_non_matching_augmentations, -1
    )

    # compute loss
    loss = loss_supervised(
        query_embeddings_adapted, document_embeddings_adapted, scores
    )

    print(loss)
    exit()

    # training
    optimizer = torch.optim.Adam(adaptor.parameters(), lr=learning_rate)
    losses = training_loop_unsupervised(
        corpus,
        queries,
        associations,
        original_model,
        adaptor,
        device,
        optimizer,
        batch_size,
        epochs,
    )

    # save model
    torch.save(adaptor.state_dict(), adaptor_file_to_save)

    # save loss curve
    plt.plot(losses, color="blue")
    plt.savefig("loss.png", dpi=300)
