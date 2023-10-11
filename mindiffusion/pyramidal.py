from typing import Dict, Tuple, List
import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
import numpy as np
from torch import Tensor


class DDPM(nn.Module):
    def __init__(
        self,
        eps_model: nn.Module,
        betas: Tuple[float, float],
        t_full: int,
        t_scaled: int,
        t_delta: int,
        sizes: list[int],
        positional_degree: int,
        batch_size: int,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super(DDPM, self).__init__()
        self.eps_model = eps_model
        self.full_scheduele = self.ddpm_schedules(betas[0], betas[1], t_full)
        self.scaled_scheduele = self.ddpm_schedules(betas[0], betas[1], t_scaled)
        self.batch_positional = self.precompute_embeddings(sizes, positional_degree)
        self.batch_positional = {
            size: pos.repeat(batch_size, 1, 1, 1)
            for size, pos in self.batch_positional.items()
        }
        self.positional = self.precompute_embeddings(sizes, positional_degree)
        self.rescaling = {
            image_size: transforms.Resize(image_size, antialias=True)
            for image_size in sizes
        }

        self.t_full = t_full
        self.t_scaled = t_scaled
        self.t_delta = t_delta
        self.sizes = sorted(sizes)
        self.min_size = self.sizes[0]
        self.criterion = criterion

    def precompute_embeddings(
        self, sizes: list[int], degree: int
    ) -> Dict[int, torch.Tensor]:
        output = {}
        for size in sizes:
            output[size] = self.get_positional_embedding(
                *self.get_pixel_coordinates(size), degree
            )
        return output

    def get_positional_embedding(self, x_matrix, y_matrix, degree):
        layers = []

        for d in range(degree):
            freq = 2**d
            layers.append(torch.sin(freq * x_matrix))
            layers.append(torch.cos(freq * x_matrix))
            layers.append(torch.sin(freq * y_matrix))
            layers.append(torch.cos(freq * y_matrix))

        return torch.stack(layers, dim=0)

    def get_pixel_coordinates(self, size):
        row = torch.arange(0, size) / (size - 1)
        x_matrix = torch.stack([row] * size)
        y_matrix = x_matrix.T
        return x_matrix, y_matrix

    def forward(self, x: torch.Tensor, should_show=False) -> torch.Tensor:
        """
        Makes forward diffusion x_t, and tries to guess epsilon value from x_t using eps_model.
        This implements Algorithm 1 in the paper.
        """
        random_size = np.random.choice(self.sizes)
        is_min_size = random_size == self.min_size
        scheduele = self.full_scheduele if is_min_size else self.scaled_scheduele
        sqrtab = scheduele["sqrtab"]
        sqrtmab = scheduele["sqrtmab"]

        x = self.rescaling[random_size](x)

        _ts = torch.randint(1, scheduele["t"] + 1, (x.shape[0],)).to(x.device)
        # t ~ Uniform(0, t_full)
        eps = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            sqrtab[_ts, None, None, None] * x + sqrtmab[_ts, None, None, None] * eps
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        positional = self.batch_positional[random_size].to(x.device)
        x_t = torch.cat((x_t, positional), dim=1)

        model_output = self.eps_model(x_t, _ts.unsqueeze(1) / scheduele["t"])

        if should_show:
            fig = plt.figure(figsize=(20, 5))
            n_images = 10
            for i in range(n_images):
                image = x_t[i]
                ax = fig.add_subplot(3, n_images, i + 1)
                ax.axis("off")
                ax.set_title("t={}".format(_ts[i]))
                ax.imshow(image.squeeze(), cmap="gray", vmin=0, vmax=1)
                ax = fig.add_subplot(3, n_images, i + 1 + n_images)
                ax.axis("off")
                ax.imshow(x[i].squeeze(), cmap="gray", vmin=0, vmax=1)
                ax = fig.add_subplot(3, n_images, i + 1 + 2 * n_images)
                ax.axis("off")
                ax.imshow(
                    model_output[i].detach().squeeze(), cmap="gray", vmin=0, vmax=1
                )
            plt.show()

        return self.criterion(eps, model_output)

    def sample_from(self, x_t: Tensor, from_t: int, scheduele, device):
        oneover_sqrta = scheduele["oneover_sqrta"]
        mab_over_sqrtmab = scheduele["mab_over_sqrtmab"]
        sqrt_beta_t = scheduele["sqrt_beta_t"]
        t_scheduele = scheduele["t"]

        for i in range(from_t, 0, -1):
            z = torch.randn_like(x_t).to(device) if i > 1 else 0
            x_t_size = x_t.shape[2]
            positional = (
                self.positional[x_t_size].repeat(x_t.shape[0], 1, 1, 1).to(device)
            )
            x_t = torch.cat((x_t, positional), dim=1)
            eps = self.eps_model(
                x_t, torch.tensor(i / t_scheduele).to(device).repeat(x_t.shape[0], 1)
            )
            x_t = (
                oneover_sqrta[i] * (x_t - eps * mab_over_sqrtmab[i])
                + sqrt_beta_t[i] * z
            )

        return x_t

    def add_noise(self, x_0, t, scheduele):
        z = torch.randn_like(x_0)
        return scheduele["sqrtab"][t] * x_0 + scheduele["sqrtmab"][t] * z

    def sample(self, n_sample: int, device) -> torch.Tensor:
        # Start by fulling denoising the smallest image
        small_x_T = torch.randn(n_sample, 1, self.min_size, self.min_size).to(device)
        x_0 = self.sample_from(small_x_T, self.t_full, self.full_scheduele, device)

        # Skip the first size, since we already sampled it
        for size in self.sizes[1:]:
            # Upscale x_0 to size
            x_0 = self.rescaling[size](x_0)
            # Find the starting t
            t_starting = int(self.t_scaled * self.t_delta)
            # Add partial noise
            x_t = self.add_noise(x_0, t_starting, self.scaled_scheduele)
            # Denoise the rest
            x_0 = self.sample_from(x_t, t_starting, self.scaled_scheduele, device)

        return x_t

    def ddpm_schedules(
        self, beta1: float, beta2: float, T: int
    ) -> Dict[str, torch.Tensor]:
        """
        Returns pre-computed schedules for DDPM sampling, training process.
        """
        assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

        beta_t = (beta2 - beta1) * torch.arange(
            0, T + 1, dtype=torch.float32
        ) / T + beta1
        sqrt_beta_t = torch.sqrt(beta_t)
        alpha_t = 1 - beta_t
        log_alpha_t = torch.log(alpha_t)
        alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

        sqrtab = torch.sqrt(alphabar_t)
        oneover_sqrta = 1 / torch.sqrt(alpha_t)

        sqrtmab = torch.sqrt(1 - alphabar_t)
        mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

        return {
            "alpha_t": alpha_t,  # \alpha_t
            "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
            "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
            "alphabar_t": alphabar_t,  # \bar{\alpha_t}
            "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
            "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
            "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
            "t": T,  # T
        }
