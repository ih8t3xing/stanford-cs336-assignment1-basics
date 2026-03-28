from torch import Tensor
from jaxtyping import Float, Int


def cross_entropy(
    inputs: Float[Tensor, "... vocab_size"], targets: Int[Tensor, "..."]
) -> Float[Tensor, ""]:
    """Compute average cross-entropy loss.

    ℓ = -log softmax(o)[x] = -(o[x] - log(sum(exp(o))))
    For numerical stability: subtract max before exp.
    """
    # Subtract max for numerical stability (log-sum-exp trick)
    max_logits = inputs.max(dim=-1, keepdim=True).values
    shifted = inputs - max_logits

    # log(sum(exp(o - max))) + max = log(sum(exp(o)))
    log_sum_exp = shifted.exp().sum(dim=-1).log() + max_logits.squeeze(-1)

    # Gather the logit at the target index
    target_logits = inputs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    # cross entropy = log_sum_exp - target_logit
    loss = log_sum_exp - target_logits

    return loss.mean()
