# Building a GPT-2 LLM From Scratch: An Educational Journey

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This repository contains a Jupyter Notebook that provides a step-by-step guide to building a GPT-2-like Large Language Model (LLM) from scratch using PyTorch. It's designed for educational purposes to demystify the core components and mechanics behind modern LLMs.

The project covers:
*   Reading and tokenizing text data (including Byte Pair Encoding with `tiktoken`).
*   Creating token embeddings and positional embeddings.
*   Implementing the self-attention mechanism (single-head, causal attention).
*   Extending to multi-head attention.
*   Building Feed-Forward Networks with GELU activation.
*   Implementing Layer Normalization and Shortcut/Residual Connections.
*   Assembling these components into a Transformer Block.
*   Constructing the full GPT model architecture.
*   Generating text with various decoding strategies (greedy, temperature scaling, top-k).
*   Setting up a training loop with loss calculation (cross-entropy, perplexity).
*   Finetuning the pre-trained GPT-2 model for a classification task (spam detection).
*   Instruction finetuning and evaluating responses using another LLM (like Llama 3 via Ollama).
*   Loading pre-trained weights from OpenAI's GPT-2.

## Table of Contents

1.  [Introduction](#introduction)
2.  [Features](#features)
3.  [Prerequisites](#prerequisites)
4.  [Installation](#installation)
5.  [Usage](#usage)
6.  [Key Components Explained](#key-components-explained)
    *   [Tokenization](#tokenization)
    *   [Embeddings](#embeddings)
    *   [Self-Attention Mechanism](#self-attention-mechanism)
    *   [Multi-Head Attention](#multi-head-attention)
    *   [Feed-Forward Network (FFN)](#feed-forward-network-ffn)
    *   [Layer Normalization](#layer-normalization)
    *   [Shortcut Connections](#shortcut-connections)
    *   [Transformer Block](#transformer-block)
    *   [GPT Model Architecture](#gpt-model-architecture)
7.  [Text Generation](#text-generation)
8.  [Training the Model](#training-the-model)
9.  [Finetuning for Classification](#finetuning-for-classification)
10. [Instruction Finetuning](#instruction-finetuning)
11. [Loading Pre-trained Weights](#loading-pre-trained-weights)
12. [Future Work](#future-work)
13. [Contributing](#contributing)
14. [License](#license)
15. [Acknowledgements](#acknowledgements)

## Introduction

This project aims to provide a clear and hands-on understanding of how GPT-style models are built from the ground up. By breaking down complex concepts into manageable Python code within a Jupyter Notebook, learners can grasp the intricacies of each component and see how they fit together to form a powerful language model.

## Features

*   **Step-by-Step Implementation:** Each core component of the Transformer architecture is built and explained.
*   **Educational Focus:** Code is commented and designed for clarity and learning.
*   **Practical Tokenization:** Uses `tiktoken` for GPT-2 compatible Byte Pair Encoding.
*   **Attention Mechanisms:** Implements causal self-attention and multi-head attention.
*   **Complete GPT Model:** Assembles transformer blocks into a full GPT-like model.
*   **Text Generation:** Includes functions for generating text with different decoding strategies.
*   **Training Loop:** Demonstrates how to train the model on a custom dataset.
*   **Finetuning Examples:**
    *   Classification finetuning on a spam detection dataset.
    *   Instruction finetuning using an Alpaca-style dataset.
*   **Pre-trained Weights:** Shows how to load and use official GPT-2 weights from OpenAI.
*   **Evaluation:** Includes methods for calculating loss, perplexity, and accuracy, plus evaluating instruction-following with another LLM.

## Prerequisites

*   Python 3.9+
*   PyTorch 2.0+
*   `tiktoken`
*   `pandas`
*   `matplotlib`
*   `tqdm`
*   `numpy`
*   `tensorflow` (for loading OpenAI's original GPT-2 weights)
*   `psutil` (for checking Ollama process in instruction finetuning evaluation)
*   (Optional) Ollama with a model like Llama 3 pulled (e.g., `ollama pull llama3`) for the instruction finetuning evaluation section.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

## Usage

Open and run the `LLM_From_Scratch.ipynb` Jupyter Notebook. The notebook is structured to be executed cell by cell, allowing you to follow the implementation and explanations sequentially.

## Key Components Explained

### Tokenization
The process of converting raw text into a sequence of smaller units called tokens. This project covers:
*   Simple regex-based tokenization.
*   Byte Pair Encoding (BPE) using OpenAI's `tiktoken` library, which is standard for GPT models.
*   Handling special tokens like `<|endoftext|>` and `<|unk|>`.

### Embeddings
*   **Token Embeddings:** Mapping discrete tokens to dense vector representations.
*   **Positional Embeddings:** Adding information about the position of tokens in a sequence, as standard attention mechanisms are permutation-invariant.

### Self-Attention Mechanism
The core of the Transformer, allowing the model to weigh the importance of different tokens in a sequence when processing a specific token.
*   **Scaled Dot-Product Attention:** The fundamental calculation involving Queries (Q), Keys (K), and Values (V).
*   **Causal Attention (Masked Self-Attention):** Ensuring that during text generation, the model only attends to previous tokens and not future ones, preventing data leakage.
*   **Dropout:** Applied to attention weights to prevent overfitting.

### Multi-Head Attention
Enhances the model's ability to focus on different parts of the input sequence simultaneously by running multiple attention mechanisms ("heads") in parallel and concatenating their outputs.

### Feed-Forward Network (FFN)
A position-wise fully connected feed-forward network applied independently to each token's representation after the attention mechanism.
*   Typically consists of two linear layers with a non-linear activation in between.
*   This project uses **GELU (Gaussian Error Linear Unit)** activation, common in GPT models.

### Layer Normalization
Applied before major operations (Pre-LayerNorm, common in modern Transformers) within each Transformer block to stabilize training by normalizing the activations across the features for each token independently.

### Shortcut Connections
Also known as residual connections, these add the input of a sub-layer (like attention or FFN) to its output. This helps in training very deep networks by mitigating the vanishing gradient problem.

### Transformer Block
The fundamental building block of GPT and other LLMs. It typically consists of:
1.  Layer Normalization
2.  Multi-Head Causal Self-Attention
3.  Shortcut Connection (+ Dropout)
4.  Layer Normalization
5.  Feed-Forward Network
6.  Shortcut Connection (+ Dropout)

### GPT Model Architecture
The full model is constructed by:
1.  Token and Positional Embeddings for the input sequence.
2.  A stack of multiple Transformer Blocks (e.g., 12 layers for GPT-2 small).
3.  A final Layer Normalization.
4.  A linear output layer that projects the final token representations to the vocabulary size to produce logits.

## Text Generation
The process of producing new text from the model given a starting prompt.
*   **Greedy Decoding:** Always picking the token with the highest probability.
*   **Temperature Scaling:** Adjusting the "peakedness" of the probability distribution. Higher temperatures lead to more randomness, lower temperatures to more deterministic outputs.
*   **Top-k Sampling:** Restricting the sampling pool to the `k` most probable next tokens.
*   Handling `<|endoftext|>` for stopping generation.

## Training the Model
*   **Data Loading:** Using PyTorch `Dataset` and `DataLoader` for efficient batching.
*   **Loss Function:** Cross-entropy loss is used to measure the difference between predicted token probabilities and actual target tokens.
*   **Perplexity:** An alternative metric (exponentiated cross-entropy) to evaluate model performance.
*   **Optimizer:** AdamW is a common choice for training Transformers.
*   **Training Loop:** Iterating through epochs and batches, calculating loss, performing backpropagation, and updating model weights.
*   **Evaluation:** Monitoring training and validation loss/accuracy.

## Finetuning for Classification
Adapting the pre-trained GPT model for a specific downstream task like text classification.
*   **Dataset:** Uses the SMS Spam Collection dataset.
*   **Modifying the Model:** Replacing the original language modeling output head with a new linear layer suited for classification (e.g., 2 outputs for spam/ham).
*   **Freezing Layers:** Optionally freezing most of the pre-trained model's weights and only training the new classification head and a few top layers.
*   **Calculating Accuracy:** Evaluating the model on a test set.

## Instruction Finetuning
Teaching the model to follow instructions and generate appropriate responses.
*   **Dataset:** Uses an Alpaca-style instruction dataset.
*   **Prompt Formatting:** Structuring inputs with 
```
"### Instruction:"
"### Input:"
"### Response:"
```
 sections.
*   **Custom Collate Function:** Handling variable-length instruction-response pairs and masking padding tokens or instruction tokens from the loss calculation.
*   **Evaluation with Another LLM:** Using a powerful LLM (like Llama 3 via Ollama) to score the finetuned model's responses based on correctness and helpfulness.

## Loading Pre-trained Weights
The notebook demonstrates how to download and load the official GPT-2 (e.g., 124M or 355M parameter versions) weights released by OpenAI into the custom `GPTModel` class. This is crucial for achieving good performance without having to pre-train from scratch on massive datasets.
*   Uses helper functions to convert TensorFlow checkpoint weights to PyTorch-compatible parameters.

## Future Work
*   Implement more advanced decoding strategies (e.g., top-p/nucleus sampling, beam search).
*   Explore different LLM architectures (e.g., Llama, Mistral).
*   Train on larger, more diverse datasets.
*   Implement parameter-efficient finetuning techniques like LoRA.

## Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
*(You'll need to create a `LICENSE` file with the MIT license text if you choose this license)*

## Acknowledgements
This project is inspired by the desire to understand LLMs from first principles and builds upon common knowledge and resources available in the deep learning community.
(If this is based on a specific book, course, or paper, it's good to mention it here).