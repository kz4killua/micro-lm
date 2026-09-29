from collections.abc import Sequence


class CharacterTokenizer:
    """A simple, character-level tokenizer."""
    def __init__(self, text: str):
        self.encoder = {c: i for i, c in enumerate(sorted(set(text)))}
        self.decoder = {i: c for c, i in self.encoder.items()}
        self.n_vocab = len(self.encoder)

    def encode(self, text: str) -> list[int]:
        return [self.encoder[c] for c in text]

    def decode(self, tokens: Sequence[int]) -> str:
        return "".join([self.decoder[t] for t in tokens])
