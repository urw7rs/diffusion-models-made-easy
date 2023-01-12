# DDIM

In DDPMs, many iterations are required to generate a single sample. 

> It takes around 20 hours to sample 50k images of size 32 x 32 from a DDPM, but less than a minute to do so from a GAN on a Nviida 2080 Ti GPU. 
> This becomes more problematic for larger images as sampling 50k images of size 256 x 256 could take nearly 1000 hours on the same GPU.
> 
> -- <cite>[Denoising Diffusion Implicit Models][1]</cite>

[1]: https://arxiv.org/abs/2010.02502

DDIM solves this problem by designing a non-Markovian diffusion process with the same training objective as DDPMs. Using this diffusion process they derive a reverse process which can sample a shorter sub-sequence to accelerate generation.


## The non-Markovian diffusion process

> Because the generative model approximates the reverse of the inference process, we need to rethink the inference process in order to reduce the number of iterations required by the generative model.
>
> -- <cite>[Denoising Diffusion Implicit Models][1]</cite>

Let us consider a family $\gQ$ of inference distributions, indexed by a real vector $\sigma \in \R_{\geq 0}^{T}$:

$$
\begin{aligned}
    q_\sigma(\vx_{1:T} | \vx_0) := q_\sigma(\vx_T | \vx_0) \prod_{t=2}^{T} q_\sigma(\vx_{t-1} | \vx_{t}, \vx_0)
\end{aligned}
$$

where $q_\sigma(\vx_{T} | \vx_0) = \gN(\sqrt{\alpha_T} \vx_0, (1 - \alpha_T) \mI)$ and for all $t > 1$,

$$
\begin{aligned}
   q_\sigma(\vx_{t-1} | \vx_t, \vx_0) = \gN\left(\sqrt{\alpha_{t-1}} \vx_{0} + \sqrt{1 - \alpha_{t-1} - \sigma^2_t} \cdot {\frac{\vx_{t}  - \sqrt{\alpha_{t}} \vx_0}{\sqrt{1 - \alpha_{t}}}}, \sigma_t^2 \mI \right).
\end{aligned}
$$

The mean function is chosen to ensure that $q_\sigma(\vx_{t} | \vx_0) = \gN(\sqrt{\alpha_t} \vx_0, (1 - \alpha_t) \mI)$ for all $t$ so that $q_\sigma(\vx_{t} | \vx_0)$ is the same as $q(\vx_t | \vx_0)$ from DDPMs. See Lemma 1 of Appendix B from the [paper](https://arxiv.org/abs/2010.02502)

The forward process can be derived from Bayes' rule:

$$
\begin{aligned}
    q_\sigma(\vx_{t} | \vx_{t-1}, \vx_0) = \frac{q_\sigma(\vx_{t-1} | \vx_{t}, \vx_0) q_\sigma(\vx_{t} | \vx_0)}{q_\sigma(\vx_{t-1} | \vx_0)},
\end{aligned}
$$

## Generative Process

Recall that $q_\sigma(\vx_{T} | \vx_0) = \gN(\sqrt{\alpha_T} \vx_0, (1 - \alpha_T) \mI)$. 
Using the reparameterization trick we can express $q_\sigma(\bx_t|\bx_0)$ as $\vx_t = \sqrt{\alpha_t}\vx_0 + \sqrt{1-\alpha_t}\epsilon$  where $\epsilon \sim \gN(\vzero, \mI)$.

Rewriting for $\vx_0$, we can estimate $\vx_0$ from $\vx$ and $\epsilon$:

$$
\begin{aligned}
    \vx_0 = \frac{\vx_t - \sqrt{1-\alpha_t}\epsilon}{\sqrt{\alpha_t}}
\end{aligned}
$$

Then we can predict the $\vx_0$ from $\vx_t$:

$$
\begin{aligned}
    f_\theta^{(t)}(\vx_t) := (\vx_t - \sqrt{1 - \alpha_t} \cdot \epsilon_{\theta}^{(t)}(\vx_t)) / \sqrt{\alpha_t}.
\end{aligned}
$$

As our reverse process is non-Markovian we can use $\vx_0$ predicted by $f_\theta^{(t)}(\vx_t)$ to sample from $q_\sigma(\vx_{t-1} | \vx_t, \vx_0)$

We can then define the generative process with a fixed prior $p_\theta(\vx_T) = \gN(\vzero, \mI)$ and

$$
\begin{aligned}
    p_\theta^{(t)}(\vx_{t-1} | \vx_t) = \begin{cases}
    \gN(f_\theta^{(1)}(\vx_1), \sigma_1^2 \mI)  & \text{if} \ t = 1 \\
    q_\sigma(\vx_{t-1} | \vx_t, f_{\theta}^{(t)}(\vx_t)) & \text{otherwise,}
    \end{cases}
\end{aligned}
$$

> We add some Gaussian noise (with covariance $\sigma_1^2 \mI$) for the case of $t = 1$ to ensure that the generative process is supported everywhere.
>
> -- <cite>[Denoising Diffusion Implicit Models][1]</cite>
