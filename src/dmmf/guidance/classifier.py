import torch
import torch.nn.functional as F

from dmme.ddpm import DDPM
from dmme.ddim import DDIM


class ClassifierMixin:
    def classifier_grad(self, classifier, y, x_t, t):
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


class ClassifierGuidedDDPM(DDPM, ClassifierMixin):
    def __init__(self, timesteps=1000, guidance_scale=10.0) -> None:
        super().__init__(timesteps)

        self.scale = guidance_scale

    def sample(self, model, classifier, y, x_t, t, noise):
        x_t = super().reverse_process(model, x_t, t, noise)
        x_t += self.scale * self.classifier_grad(classifier, y, x_t, t)

        return x_t


class ClassifierGuidedDDIM(DDIM, ClassifierMixin):
    def __init__(
        self, timesteps, tau_schedule="quadratic", guidance_scale=10.0
    ) -> None:
        super().__init__(timesteps, tau_schedule)

        self.scale = guidance_scale

    def reverse_process(self, model, classifier, y, x_t, t):
        alpha_bar_t_minus_one = self.alpha_bar[t - 1]
        alpha_bar_t = self.alpha_bar[t]

        grad = self.classifier_grad(classifier, y, x_t, t)
        epsilon = model(x_t, t) - torch.sqrt(1 - alpha_bar_t) * self.scale * grad

        x_t = (
            torch.sqrt(alpha_bar_t_minus_one)
            * (x_t - torch.sqrt(1 - alpha_bar_t) * epsilon)
            / torch.sqrt(alpha_bar_t)
        ) + torch.sqrt(1 - alpha_bar_t_minus_one) * epsilon

        return x_t

    def sample(self, model, classifier, y, x_t, t):
        return self.reverse_process(model, classifier, y, x_t, t)
