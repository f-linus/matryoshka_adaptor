import torch
import torch.nn.functional as F


def unsupervised_loss(
    embeddings_original: torch.tensor,
    embeddings_adapted: torch.tensor,
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


def rank_loss(
    query_embeddings_adapted: torch.tensor,
    document_embeddings_adapted: torch.tensor,
    scores_batched: torch.tensor,
):
    device = query_embeddings_adapted.device
    batch_size = query_embeddings_adapted.shape[0]
    max_embedding_dim = query_embeddings_adapted.shape[1]

    loss_sum = 0

    # we iterate over the batch (i.e. all queries)
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