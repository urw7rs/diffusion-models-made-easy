from torch.nn.functional import mse_loss


def simple_loss(noise, estimated_noise):
    r"""Simple Loss objective :math:`L_\text{simple}`, MSE loss between noise and predicted noise

    Args:
        noise (torch.Tensor): noise used in the forward process
        estimated_noise (torch.Tensor): estimated noise with the same shape as :code:`noise`

    """
    return mse_loss(noise, estimated_noise)
