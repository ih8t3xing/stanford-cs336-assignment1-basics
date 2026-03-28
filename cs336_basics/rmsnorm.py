import torch.nn as nn
import torch 

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape
        # Compute the mean squared value of the input tensor along the last dimension
        in_dtype = x.dtype
        x = x.to(torch.float32)
        mean_squared = torch.mean(x ** 2, dim=-1, keepdim=True)
        # Normalize the input tensor by dividing it by the square root of the mean squared value plus epsilon
        normalized_x = x / torch.sqrt(mean_squared + self.eps)
        # Scale the normalized tensor by multiplying it with the learnable weight parameter
        output = torch.einsum("...i,i->...i", normalized_x, self.weight)
        return output.to(in_dtype)