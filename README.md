# Micro LM
A simple transformer language model, implemented from scratch in PyTorch. The model is based on architecture described in [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

## Setup
1. Clone this repository to your local machine. 
2. Install uv using the instructions [here](https://docs.astral.sh/uv/getting-started/installation/).
3. Navigate to the project directory in your terminal.
3. Install dependencies using `uv sync`.

## Training
The `train` command trains a language model on a single text file and saves the trained model. The basic form is:
```
uv run main.py train path/to/file.txt path/to/model.pt
```

The following options are available:
| Option | Type | Default | Description |
| --- | --- | ---: | --- |
| `--train-fraction` | `float` | `0.9` | The fraction of the input tokens used for training. The remainder is used for validation. |
| `--steps` | `int` | `10000` | The number of training steps. |
| `--batch-size` | `int` | `12` | The number of sequences used in each training batch. |
| `--learning-rate` | `float` | `0.001` | The learning rate used by the optimizer. |
| `--context-size` | `int` | `64` | The maximum number of tokens used as context during training and generation. |
| `--d-model` | `int` | `128` | The dimensionality of token representations in the model, including token embeddings and positional encodings. |
| `--n-layers` | `int` | `4` | The number of transformer layers. |
| `--h` | `int` | `4` | The number of attention heads. |
| `--d-k` | `int` | `32` | The dimensionality of the attention keys and queries used in each attention head. |
| `--d-v` | `int` | `32` | The dimensionality of the attention values used in each attention head. |
| `--d-ff` | `int` | `512` | The dimensionality of the position-wise feed-forward network. |
| `--seed` | `int` | `42` | The seed used for random number generation. |

## Generation
The `generate` command loads a trained model and completes an input prompt. The basic form is:
```
uv run main.py generate path/to/model.pt "<prompt>"
```

The following options are available:
| Option | Type | Default | Description |
| --- | --- | ---: | --- |
| `--max-tokens` | `int` | `1000` | The maximum number of tokens to generate. |
| `--temperature` | `float` | `1.0` | Controls the randomness of token sampling. Lower values make generation more deterministic. Higher values make it more random. |
| `--seed` | `int` | `42` | The seed used for random number generation. |
