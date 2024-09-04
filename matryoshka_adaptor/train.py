import logging

import datasets
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from matryoshka_adaptor.losses import rank_loss, unsupervised_loss
from matryoshka_adaptor.model import MatryoshkaAdaptor

logger = logging.getLogger(__name__)


def training_loop(
    corpus: datasets.Dataset,
    queries: datasets.Dataset,
    associations: datasets.Dataset,
    original_model: SentenceTransformer,
    adaptor_model: MatryoshkaAdaptor,
    n_epochs_unsupervised: int = 5,
    n_epochs_supervised: int = 5,
    query_batch_size: int = 1024,
    corpus_batch_size: int = 1024,
    n_non_matching_augmentations: int = 9,
    log_interval: int = 10,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    encoding_batch_size: int = 512,
    unsupervised_learning_model_file="adaptor_unsupervised.pt",
    supervised_learning_model_file="adaptor_supervised.pt",
    loss_curve_file="loss_curve.png",
    learning_rate_unsupervised: float = 0.001,
    learning_rate_supervised: float = 0.001,
    gamma=100,
):
    loss_trajectory_unsupervised = []
    loss_trajectory_supervised = []

    optimizer = torch.optim.Adam(
        adaptor_model.parameters(), lr=learning_rate_unsupervised
    )

    # unsupervised training
    step = 0
    for epoch in range(n_epochs_unsupervised):
        for corpus_embeddings, _, _, _ in batch_iterator(
            corpus=corpus,
            queries=queries,
            associations=associations,
            original_model=original_model,
            query_batch_size=0,  # set to 0 to only use unsupervised corpus data
            corpus_batch_size=corpus_batch_size,
            n_non_matching_augmentations=n_non_matching_augmentations,
            device=device,
            encoding_batch_size=encoding_batch_size,
        ):
            adaption_delta = adaptor_model.forward(corpus_embeddings)
            corpus_embeddings_new = corpus_embeddings + adaption_delta

            # compute loss
            loss = unsupervised_loss(corpus_embeddings, corpus_embeddings_new)
            loss_trajectory_unsupervised.append(loss.item())

            # optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log
            if step % log_interval == 0:
                logger.info(
                    f"Unsupervised epoch {epoch+1}/{n_epochs_unsupervised}, step {step}: Loss: {loss.item()}"
                )
            step += 1

    # save model
    if n_epochs_unsupervised > 0:
        torch.save(adaptor_model.state_dict(), unsupervised_learning_model_file)
        logger.info(f"Unsupervised model saved to {unsupervised_learning_model_file}")

    # reset optimizer
    optimizer = torch.optim.Adam(
        adaptor_model.parameters(), lr=learning_rate_supervised
    )

    # supervised training
    step = 0
    for epoch in range(n_epochs_supervised):
        for (
            corpus_embeddings,
            query_embeddings,
            document_embeddings,
            scores,
        ) in batch_iterator(
            corpus=corpus,
            queries=queries,
            associations=associations,
            original_model=original_model,
            query_batch_size=query_batch_size,
            corpus_batch_size=corpus_batch_size,
            n_non_matching_augmentations=n_non_matching_augmentations,
            device=device,
            encoding_batch_size=encoding_batch_size,
        ):
            # adapt unsupervised corpus embeddings as before
            adaption_delta = adaptor_model.forward(corpus_embeddings)
            corpus_embeddings_new = corpus_embeddings + adaption_delta

            # adapt supervised embedding data
            query_embeddings_adapted = adaptor_model.forward(query_embeddings)
            document_embeddings_adapted = adaptor_model.forward(document_embeddings)

            # loss
            rank_loss_value = gamma * rank_loss(
                query_embeddings_adapted, document_embeddings_adapted, scores
            )
            unsupervised_loss_value = unsupervised_loss(
                corpus_embeddings, corpus_embeddings_new
            )

            loss = rank_loss_value + unsupervised_loss_value
            loss_trajectory_supervised.append(loss.item())

            # optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log
            if step % log_interval == 0:
                logger.info(
                    f"Supervised epoch {epoch+1}/{n_epochs_supervised}, step {step}: Loss: {loss.item()}, (rank: {rank_loss_value.item()}, unsupervised: {unsupervised_loss_value.item()})"
                )

            step += 1

    # save model
    torch.save(adaptor_model.state_dict(), supervised_learning_model_file)
    logger.info(f"Supervised model saved to {supervised_learning_model_file}")

    # save loss curve
    _, ax = plt.subplots()
    ax.plot(loss_trajectory_unsupervised + loss_trajectory_supervised, color="blue")

    if len(loss_trajectory_unsupervised) > 0:
        ax.axvline(x=len(loss_trajectory_unsupervised), color="red", linestyle="--")
    plt.savefig(loss_curve_file, dpi=300)

    return loss_trajectory_unsupervised, loss_trajectory_supervised


def batch_iterator(
    corpus: datasets.Dataset,
    queries: datasets.Dataset,
    associations: datasets.Dataset,
    original_model: SentenceTransformer,
    query_batch_size: int = 1024,  # set to 0 to only use unsupervised corpus data
    corpus_batch_size: int = 1024,
    n_non_matching_augmentations: int = 9,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    encoding_batch_size: int = 512,
):
    # shuffle queries and corpus
    queries = queries.shuffle()
    corpus = corpus.shuffle()

    # if query batch size > 0, create dataframes to efficiently select pairs
    if query_batch_size > 0:
        associations_df = associations.to_pandas()
        corpus_df = corpus.to_pandas()
        queries_df = queries.to_pandas()

        corpus_df.set_index("_id", inplace=True)
        queries_df.set_index("_id", inplace=True)

        associations_df = associations_df.join(corpus_df, on="corpus-id", how="left")
        associations_df = associations_df.join(
            queries_df, on="query-id", how="left", rsuffix="_query"
        )

    corpus_batch_ptr = 0
    query_batch_ptr = 0
    while True:
        if corpus_batch_ptr + corpus_batch_size > len(corpus):
            break

        if (
            query_batch_ptr + query_batch_size > len(associations)
            and query_batch_size > 0
        ):
            break

        query_embeddings = None
        document_embeddings = None
        scores = None
        if query_batch_size > 0:
            query_batch = associations_df.iloc[
                query_batch_ptr : query_batch_ptr + query_batch_size
            ]

            # we add non matching documents to the queries for somewhat contrastive training
            # in theory, documents added with score 0 could actually be part of the matching ones
            # however, for sufficiently large corpi prob. is -> 0 and we prefer to not do the extra computation expenditure
            zero_score_augmentations = pd.DataFrame(
                {
                    "text_query": query_batch["text_query"].to_list()
                    * n_non_matching_augmentations,
                    "text": associations_df["text"]
                    .sample(
                        n=len(query_batch) * n_non_matching_augmentations,
                        replace=True,
                    )
                    .to_list(),
                }
            )
            zero_score_augmentations["score"] = 0

            batch_augmented = pd.concat([query_batch, zero_score_augmentations])

            query_embeddings_np = original_model.encode(
                query_batch["text_query"].to_list(),
                batch_size=encoding_batch_size,
                show_progress_bar=False,
            )
            document_embeddings_np = original_model.encode(
                batch_augmented["text"].to_list(),
                batch_size=encoding_batch_size,
                show_progress_bar=False,
            )
            scores = batch_augmented["score"].to_list()

            # transform into torch Tensor
            query_embeddings = torch.tensor(query_embeddings_np, device=device)
            document_embeddings = torch.tensor(document_embeddings_np, device=device)
            scores = torch.tensor(scores, device=device)

            # reshaping and re-ordering
            document_embeddings = document_embeddings.reshape(
                1 + n_non_matching_augmentations, query_batch_size, -1
            )
            document_embeddings = document_embeddings.transpose(0, 1)

            scores = scores.reshape(1 + n_non_matching_augmentations, query_batch_size)
            scores = scores.transpose(0, 1)

        # unsupervised corpus data
        corpus_batch = corpus[corpus_batch_ptr : corpus_batch_ptr + corpus_batch_size][
            "text"
        ]

        embeddings_np = original_model.encode(
            corpus_batch, batch_size=encoding_batch_size, show_progress_bar=False
        )
        corpus_embeddings = torch.tensor(embeddings_np, device=device)

        # move pointers forward
        corpus_batch_ptr += corpus_batch_size
        query_batch_ptr += query_batch_size

        # yield batch
        yield corpus_embeddings, query_embeddings, document_embeddings, scores
