# DDIM

In DDPMs, many iterations are required to generate a single sample. 

> It takes around 20 hours to sample 50k images of size 32 x 32 from a DDPM, but less than a minute to do so from a GAN on a Nviida 2080 Ti GPU. 
> This becomes more problematic for larger images as sampling 50k images of size 256 x 256 could take nearly 1000 hours on the same GPU.
> 
> -- <cite>[Denoising Diffusion Implicit Models][1]</cite>

[1]: https://arxiv.org/abs/2010.02502

DDIM solves this problem by designing a non-Markovian diffusion process with the same training objective as DDPMs. Using this diffusion process they derive a reverse process which can sample a shorter sub-sequence to accelerate generation.
