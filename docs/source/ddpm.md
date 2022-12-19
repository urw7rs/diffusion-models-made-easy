# DDPM

In physics and chemistry, the microscopic reversibility states that 

> "the microscopic detailed dynamics of particles and fields is time-reversible because the microscopic equations of motion are symmetric with respect to inversion in time"

This mean that if a data distribution is diffused to noise, the reverse process exists in a microscopic level.

This is because the equations that describe the dynamics are "symmetric with respect to inversion in time".

Assuming this reverse process exists, the Denoising Diffusion Probabilistic Model generates data by gradually denoising data starting from Gaussian noise.

Since this principle holds for "microscopic detailed dynamics", we design a Forward Diffusion process that gradually diffuses data to Gaussian noise.

In each step, we sample from a Gaussian distribution that perturbs the data. Formally, we define it as a Markov chain of Gaussians:

$$
\begin{aligned}
q(\bx_{1:T} | \bx_0) &\defeq \prod_{t=1}^T q(\bx_t | \bx_{t-1} ), \qquad 
q(\bx_t|\bx_{t-1}) \defeq \mathcal{N}(\bx_t;\sqrt{1-\beta_t}\bx_{t-1},\beta_t \bI)
\end{aligned}
$$

Note that we can sample $\bx_t$ for an arbitrary timestep $t# in closed form:

$$
\begin{aligned}
\alpha_t &\defeq 1-\beta_t, \quad \bar\alpha_t \defeq \prod_{s=1}^t \alpha_s \\
q(\bx_t|\bx_0) &= \mathcal{N}(\bx_t; \sqrt{\bar\alpha_t}\bx_0, (1-\bar\alpha_t)\bI)
\end{aligned}
$$

If $\beta_t$ is small enough, the reverse process should also exist. And since the process is symmetric it should also be a Markov chain of Gaussians starting from $p(\bx_T)=\mathcal{N}(\bx_T; \bzero, \bI)$:

$$
\begin{aligned}
  p_\theta(\bx_{0:T}) &\defeq p(\bx_T)\prod_{t=1}^T p_\theta(\bx_{t-1}|\bx_t), \qquad 
  p_\theta(\bx_{t-1}|\bx_t) \defeq \mathcal{N}(\bx_{t-1}; \bmu_\theta(\bx_t, t), \bSigma_\theta(\bx_t, t))
\end{aligned}
$$

In order to generate data, we sample from the Standard Normal distribution then iteratively sample $p_\theta(x_{t-1}|x_t)$

For training, we optimize the variance lower bound objective from variational autoencoders.

$$
\begin{aligned}
\Ea{-\log p_\theta(\bx_0)} &\leq \Eb{q}{ - \log \frac{p_\theta(\bx_{0:T})}{q(\bx_{1:T} | \bx_0)}}
  \\
&= \mathbb{E}_q\bigg[ -\log p(\bx_T) - \sum_{t \geq 1} \log \frac{p_\theta(\bx_{t-1} | \bx_t)}{q(\bx_t|\bx_{t-1})} \bigg] \eqqcolon L
\end{aligned}
$$

We can reparameterize the variance lower bound into

$$
\begin{aligned}
\mathbb{E}_q \bigg[ \underbrace{\kl{q(\bx_T|\bx_0)}{p(\bx_T)}}_{L_T \, \approx \, 0} + \sum_{t > 1} \underbrace{\kl{q(\bx_{t-1}|\bx_t,\bx_0)}{p_\theta(\bx_{t-1}|\bx_t)}}_{L_{t-1}} \underbrace{-\log p_\theta(\bx_0|\bx_1)}_{L_0, \, \text{ignore}} \bigg]
\end{aligned}
$$

Rewriting loss as $L = L_T + \sum_{t\lt1}L_{t-1} + L_0$

$$
\begin{aligned}
q(\bx_{t-1}|\bx_t,\bx_0) &=  \mathcal{N}(\bx_{t-1}; \tilde\bmu_t(\bx_t, \bx_0), \tilde\beta_t \bI), \\
\text{where}\quad \tilde\bmu_t(\bx_t, \bx_0) &\defeq \frac{\sqrt{\bar\alpha_{t-1}}\beta_t }{1-\bar\alpha_t}\bx_0 + \frac{\sqrt{\alpha_t}(1- \bar\alpha_{t-1})}{1-\bar\alpha_t} \bx_t \quad \text{and} \quad
\tilde\beta_t \defeq \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t
\end{aligned}
$$

We parameterize the neural network to closely match the forward process in $L_{t-1}$

Recall that $p_\theta(\bx_{t-1}|\bx_t) = \mathcal{N}(\bx_{t-1}; \bmu_\theta(\bx_t, t), \bSigma_\theta(\bx_t, t))$ for ${1 \lt t \leq T}$.

With $p_\theta(\bx_{t-1} | \bx_t) = \mathcal{N}(\bx_{t-1}; \bmu_\theta(\bx_t, t), \sigma_t^2\bI)$, we can write:

> Experimentally, both $\sigma_t^2 = \beta_t$ and $\sigma_t^2 = \tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$ had similar results.

$$
\begin{aligned}
  L_{t-1}
   &= \mathbb{E}_q \bigg[ \frac{1}{2\sigma_t^2} \|\tilde\mu_t(x_t,x_0) - \mu_\theta(x_t, t)\|^2 \bigg] + C \\
\tilde\mu(x_t,t) &= \frac{1}{\sqrt{1-\beta_t}}\bigg(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon(x_t,t)\bigg) \\
\mu_\theta(x_t,t) &= \frac{1}{\sqrt{1-\beta_t}}\bigg(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\bigg)
\end{aligned}
$$

Input image data is assumed to be integers in ${0, 1, \, ... \, ,255}$ scaled linearly to $[-1, 1]$. The last step of the reverse process is set to an independent discrete decoder. At the final step of sampling, noise is not used.

Then we can simplify the loss to

$$
\begin{aligned}
\E_{\bx_0, \bepsilon}\bigg[ \underbrace{\frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1-\bar\alpha_t)}}_{\lambda_t}  \left\| \bepsilon - \bepsilon_\theta(\sqrt{\bar\alpha_t} \bx_0 + \sqrt{1-\bar\alpha_t}\bepsilon, t) \right\|^2 \bigg]
\end{aligned}
$$

For small $t$, $\lambda_t$ is too large, In the paper setting $\lambda_t = 1$ improves sample quality

$$
\begin{aligned}
 L_\mathrm{simple} &\defeq \E_{t \sim \mathcal{U}(1, T), \bx_0, \bepsilon}\big[ \| \bepsilon - \bepsilon_\theta(\underbrace{\sqrt{\bar\alpha_t} \bx_0 + \sqrt{1-\bar\alpha_t}\bepsilon}_{\bx_t}, t) \|^2 \big] \\
\end{aligned}
$$


```{eval-rst}
.. currentmodule:: dmme

.. autosummary::
    :nosignatures:

    dmme.DDPMSampler
    dmme.ddpm.linear_schedule
    dmme.ddpm.UNet
    dmme.LitDDPM
```

## Sampler

```{eval-rst}
.. currentmodule:: dmme

.. autoclass:: DDPMSampler
    :members:
```

```{eval-rst}
.. currentmodule:: dmme.ddpm

.. autofunction:: linear_schedule
```

## Model

```{eval-rst}
.. currentmodule:: dmme.ddpm

.. autoclass:: UNet
    :members:
```

```{eval-rst}
.. currentmodule:: dmme.ddpm.unet

.. autosummary::

    TimeStepEmbedding
    SinusoidalPositionEmbeddings
    Block
    DownSample
    UpSample
    Attention
    ResBlock
    conv2d
```

```{eval-rst}
.. currentmodule:: dmme.ddpm.unet

.. autoclass:: TimeStepEmbedding
    :members:

.. autoclass:: SinusoidalPositionEmbeddings
    :members:

.. autoclass:: Block
    :members:

.. autoclass:: DownSample
    :members:

.. autoclass:: UpSample
    :members:

.. autoclass:: Attention
    :members:

.. autoclass:: ResBlock
    :members:

.. autofunction:: conv2d
```

## Training

``` {eval-rst}
.. currentmodule:: dmme

.. autoclass:: LitDDPM
    :members:
```
