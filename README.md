# Micro LM
A simple Transformer language model, implemented from scratch in PyTorch. The model is based on the architecture described in [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

## Example
Here's some sample output from a small model trained on `corpus/shakespeare.txt`.

```
ROMEO:
As thou art day?

PETER:
Yes, sir, and you charge your good will his comploted with our pleasure?

BUCKINGHAM:
I come the prince's and well-leave, and my king?

HENRY BOLINGBROKE:
Thanks, it with my soul stain thine own and
To be move underceive my differmiors with chivilen.

HENRY BOLINGBROKE:
O God! the was fast for seven passing for swafe
Being to the city see mired to the wool,
And did seven in the want purged out the air.

DERBY:
I twice it not make before it.
```

Not super coherent, but it sounds like Shakespeare. You may get better results with more training data or a larger model.

## Setup
1. Clone this repository to your local machine. 
2. Install uv using the instructions [here](https://docs.astral.sh/uv/getting-started/installation/).
3. Navigate to the project directory in your terminal.
4. Install dependencies using `uv sync`.

## Training
The `train` command trains a language model on a single text file and saves the trained model. The basic form is:
```
uv run main.py train path/to/file.txt path/to/model.pt
```

Some sample text files are included in `corpus/`. 

The following options are available:

| Option | Type | Default | Description |
| --- | --- | ---: | --- |
| `--train-fraction` | `float` | `0.9` | The fraction of the input tokens used for training. The remainder is used for validation. |
| `--steps` | `int` | `5000` | The number of training steps. |
| `--batch-size` | `int` | `12` | The number of sequences used in each training batch. |
| `--learning-rate` | `float` | `0.001` | The learning rate used by the optimizer. |
| `--context-size` | `int` | `64` | The maximum number of tokens used as context during training and generation. |
| `--d-model` | `int` | `128` | The dimensionality of token representations in the model, including token embeddings and positional encodings. |
| `--n-layers` | `int` | `4` | The number of Transformer layers. |
| `--h` | `int` | `4` | The number of attention heads. |
| `--d-k` | `int` | `32` | The dimensionality of the attention keys and queries used in each attention head. |
| `--d-v` | `int` | `32` | The dimensionality of the attention values used in each attention head. |
| `--d-ff` | `int` | `512` | The dimensionality of the position-wise feed-forward network. |
| `--seed` | `int` | `42` | The seed used for random number generation. |

The default options are intentionally small and should be fine for most consumer devices. You can scale these up or down depending on the amount of compute you have available.

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
