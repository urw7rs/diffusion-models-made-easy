from torch import nn, Tensor
import torch.nn.functional as F


class SimpleLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, noise: Tensor, noise_estimate: Tensor):
        r"""Computes the loss

        :math:`L_\text{simple} = \mathbb{E}_{\bold{x}_0\sim q(\bold{x}_0),
        \epsilon\sim\mathcal{N}(\bold{0},\bold{I}),
        t\sim\mathcal{U}(1,T)}
        \left[\|\epsilon-\epsilon_\theta(\bold{x}_t, t) \|^2\right]`

        Args:
            model (nn.Module): model for estimating noise
            x_0 (torch.Tensor): :math:`x_0`
            t (int, optional): sampled :math:`t`
            noise (torch.Tensor, optional): sampled :math:`\epsilon`
        """

        loss = F.mse_loss(noise, noise_estimate)
        return loss
