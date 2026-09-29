from pathlib import Path

import torch
import typer

from generate import generate_completion
from tokenizer import CharacterTokenizer
from train import train_language_model, train_val_split
from transformer import TransformerLanguageModel

app = typer.Typer()


@app.command()
def train(
    text_path: Path,
    save_path: Path,
    *,
    train_fraction: float = 0.9,
    steps: int = 10_000,
    batch_size: int = 12,
    learning_rate: float = 1e-3,
    context_size: int = 64,
    d_model: int = 128,
    n_layers: int = 4,
    h: int = 4,
    d_k: int = 32,
    d_v: int = 32,
    d_ff: int = 512,
    seed: int = 42,
):
    torch.manual_seed(seed)

    text = text_path.read_text(encoding="utf-8")
    tokenizer = CharacterTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_data, val_data = train_val_split(data, train_fraction)

    model = TransformerLanguageModel(
        n_vocab=tokenizer.n_vocab,
        context_size=context_size,
        d_model=d_model,
        n_layers=n_layers,
        h=h,
        d_k=d_k,
        d_v=d_v,
        d_ff=d_ff,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    train_language_model(
        model=model,
        optimizer=optimizer,
        train_data=train_data,
        val_data=val_data,
        context_size=context_size,
        batch_size=batch_size,
        steps=steps,
    )

    checkpoint = {
        "model_config": {
            "n_vocab": tokenizer.n_vocab,
            "context_size": context_size,
            "d_model": d_model,
            "n_layers": n_layers,
            "h": h,
            "d_k": d_k,
            "d_v": d_v,
            "d_ff": d_ff,
        },
        "model_state_dict": model.state_dict(),
        "tokenizer_chars": "".join(tokenizer.encoder),
    }
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, save_path)


@app.command()
def generate(load_path: Path, prompt: str, max_tokens: int = 1_000, seed: int = 42):
    checkpoint = torch.load(load_path, map_location="cpu", weights_only=True)

    model = TransformerLanguageModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])

    tokenizer = CharacterTokenizer(checkpoint["tokenizer_chars"])

    torch.manual_seed(seed)

    completion = generate_completion(
        model=model,
        tokenizer=tokenizer,
        context_size=checkpoint["model_config"]["context_size"],
        prompt=prompt,
        max_tokens=max_tokens,
    )

    typer.echo(completion)


if __name__ == "__main__":
    app()
