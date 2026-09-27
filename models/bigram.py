from torch import Tensor, nn


class BigramLanguageModel(nn.Module):
    """A simple language model where each next token is a function of only the previous token."""

    def __init__(self, n_vocab: int):
        super().__init__()
        self.embeddings = nn.Embedding(num_embeddings=n_vocab, embedding_dim=n_vocab)

    def forward(self, x: Tensor) -> Tensor:
        return self.embeddings(x)
