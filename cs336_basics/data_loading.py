import numpy as np
import torch


def get_batch(
    dataset: np.ndarray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    # Sample random starting indices; each sequence needs context_length+1 tokens
    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    # print(f"starts: {starts}")
    x = np.stack([dataset[i : i + context_length] for i in starts])
    y = np.stack([dataset[i + 1 : i + context_length + 1] for i in starts])
    x = torch.tensor(x.astype(np.int32), dtype=torch.long, device=device)
    y = torch.tensor(y.astype(np.int32), dtype=torch.long, device=device)
    # print(f"x: {x.shape}, y: {y.shape}")
    return x, y
