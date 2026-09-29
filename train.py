import torch
import torch.nn.functional as F
from torch import Tensor, nn


def train_language_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: Tensor,
    val_data: Tensor,
    context_size: int,
    batch_size: int,
    steps: int,
):
    for step in range(steps):
        model.train()

        batch_x, batch_y = sample_sequence_batch(train_data, context_size, batch_size)
        logits = model(batch_x)
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.reshape(B * T, C), batch_y.reshape(B * T))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            val_loss = estimate_loss(model, val_data, context_size, batch_size)
            print(f"step {step}: {val_loss}")


@torch.no_grad()
def estimate_loss(model: nn.Module, data: Tensor, context_size: int, batch_size: int):
    model.eval()

    losses = []
    for _ in range(100):
        batch_x, batch_y = sample_sequence_batch(data, context_size, batch_size)
        logits = model(batch_x)
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.reshape(B * T, C), batch_y.reshape(B * T))

        losses.append(loss.item())

    return sum(losses) / len(losses)


def sample_sequence_batch(
    tokens: Tensor, sequence_length: int, batch_size: int
) -> tuple[Tensor, Tensor]:
    starts = torch.randint(
        0, len(tokens) - sequence_length, (batch_size,), device=tokens.device
    )
    offsets = torch.arange(sequence_length, device=tokens.device)
    indices = starts[:, None] + offsets[None, :]
    x = tokens[indices]
    y = tokens[indices + 1]
    return x, y


def train_val_split(data: Tensor, train_fraction: float) -> tuple[Tensor, Tensor]:
    i = int(train_fraction * len(data))
    return data[:i], data[i:]
