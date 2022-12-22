from dmme.ddpm import DDPMSampler


class DDIMSampler(DDPMSampler):
    """Wrapper for computing forward and reverse processes,
    sampling data, and computing loss for DDIM

    > Implements sampling from an implicit model that is trained with the same procedure as Denoising Diffusion Probabilistic Model, but costs much less time and compute if you want to sample from it

    Paper: https://arxiv.org/abs/2010.02502

    Code: https://github.com/ermongroup/ddim

    Args:
        model (nn.Module): model
        timesteps (int): diffusion timesteps
        sigma (float): :math:`sigma_t` wich controls characteristics of the generative process
    """
