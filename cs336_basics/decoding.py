from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


def decode(
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: Optional[torch.device] = None,
) -> str:
    """
    Generate a completion for a prompt using the language model.

    Args:
        model: The language model (TransformerLM).
        tokenizer: Tokenizer with encode/decode methods.
        prompt: The input text prompt.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Softmax temperature. Values < 1.0 sharpen the distribution,
                     values > 1.0 flatten it. Must be > 0.
        top_p: Nucleus sampling threshold. Only sample from the smallest set of
               tokens whose cumulative probability >= top_p. Set to 1.0 to disable.
        device: Device to run inference on.

    Returns:
        The generated completion string (not including the prompt).
    """
    if device is None:
        device = next(model.parameters()).device

    # Get the end-of-text token id
    eot_token = "<|endoftext|>"
    eot_id = tokenizer.encode(eot_token)
    # eot_id should be a single token; take the last one if encode returns a list
    if isinstance(eot_id, list):
        eot_id = eot_id[-1]

    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    else:
        input_ids = input_ids.to(device).unsqueeze(0) if input_ids.dim() == 1 else input_ids.to(device)

    generated_ids = []

    model.eval()
    with torch.no_grad():
        tokens = input_ids  # (1, seq_len)

        for _ in range(max_new_tokens):
            logits = model(tokens)  # (1, seq_len, vocab_size)
            next_logits = logits[0, -1, :]  # (vocab_size,)

            # Apply temperature scaling
            if temperature != 1.0:
                next_logits = next_logits / temperature

            # Compute probabilities
            probs = torch.softmax(next_logits, dim=-1)

            # Top-p (nucleus) sampling
            if top_p < 1.0:
                # Sort in descending order
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                # Remove tokens once cumulative probability exceeds top_p
                # Keep at least one token
                sorted_indices_to_remove = cumulative_probs - sorted_probs > top_p
                sorted_probs[sorted_indices_to_remove] = 0.0
                # Renormalize
                sorted_probs = sorted_probs / sorted_probs.sum()

                # Sample from filtered distribution
                sampled_rank = torch.multinomial(sorted_probs, num_samples=1)
                next_token = sorted_indices[sampled_rank]
            else:
                next_token = torch.multinomial(probs, num_samples=1)  # (1,)

            next_token_id = next_token.item()
            generated_ids.append(next_token_id)

            # Stop if we hit end-of-text
            if next_token_id == eot_id:
                break

            # Append to sequence
            tokens = torch.cat([tokens, next_token.unsqueeze(0).to(device)], dim=1)

    return tokenizer.decode(generated_ids)
