import torch
from torch import nn
import torch.nn.functional as F


def classifier_grad(
    classifier: nn.Module, y: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
):
    x_t = x_t.requires_grad_(True)
    t = t.float().requires_grad_(True)

    with torch.enable_grad():
        logits = classifier(x_t, t)
        log_probs = F.log_softmax(logits, dim=1)
        log_probs_of_y = log_probs[:, y]
        # sum to backpropagate, will preserve gradients per batch
        (grad_log_probs_of_y,) = torch.autograd.grad(log_probs_of_y.sum(), x_t)

    x_t = x_t.detach().requires_grad_(False)
    t = t.detach().requires_grad_(False)

    return grad_log_probs_of_y
