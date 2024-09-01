# Matryoshka Adaptor

Implementation of the Matryoshka Adaptor ([paper](https://arxiv.org/abs/2407.20243)).

The Matryoshka Adaptor is a proposed method to tune an adaptor-network to map embeddings from some source (potentially black-box) embedding model into a different space with a more desirable structure. First and foremost, mapped embeddings should posses the "Matryoshka property" (see [here](https://huggingface.co/blog/matryoshka)) which in embedding models refers to the property of largely maintained performance when only the first $k$ dimensions of mapped embeddings are considered. Use case: allows performance trade-offs through dim. reduction that can be very desirable.

For details of the method please refer to the paper.

## Usage



## References

Yoon, Jinsung et al. “Matryoshka-Adaptor: Unsupervised and Supervised Tuning for Smaller Embedding Dimensions.” ArXiv abs/2407.20243 (2024): n. pag. [ArXiv](https://arxiv.org/abs/2407.20243)

