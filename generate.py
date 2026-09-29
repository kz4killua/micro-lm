import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tokenizer import CharacterTokenizer


@torch.no_grad()
def generate_completion(
    model: nn.Module,
    tokenizer: CharacterTokenizer,
    prompt: str,
    max_tokens: int,
    context_size: int,
    temperature: float,
) -> str:
    model.eval()

    completion = []

    device = next(model.parameters()).device
    x = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).view(
        1, -1
    )
    for _ in range(max_tokens):
        y = generate_token(model, x, context_size, temperature)
        completion.append(int(y.item()))
        x = torch.cat((x, y), dim=1)[:, -context_size:]

    return prompt + tokenizer.decode(completion)


@torch.no_grad()
def generate_token(
    model: nn.Module, x: Tensor, context_size: int, temperature: float
) -> Tensor:
    context = x[:, -context_size:]
    logits = model(context)[:, -1, :]
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1)
