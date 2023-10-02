from typing import Dict, Tuple
import torch
import torch.nn as nn
from torchvision import transforms


class Pixelate:
    def __init__(self, sizes: list[int], n_between: int = 1):
        """Sizes is a list of ints from smallest to largest"""
        self.sizes = sizes
        self.transforms = [self.set_image_to_random_grey]
        interpolation = transforms.InterpolationMode.NEAREST
        for size in sizes:
            self.transforms.append(
                transforms.Compose(
                    [
                        transforms.Resize(size, interpolation),
                        transforms.Resize(sizes[-1], interpolation),
                    ]
                )
            )
        self.n_between = n_between
        self.n_transforms = len(self.transforms)
        self.T = self.n_transforms + (self.n_transforms - 1) * self.n_between

    def seed(self, seed: int):
        torch.manual_seed(seed)

    def get_xT(self):
        """Returns a random grey image of max size"""
        size = self.sizes
        color = torch.rand(1)
        return torch.ones(1, size, size) * color

    def set_image_to_random_grey(self, image: torch.Tensor):
        return image * 0 + torch.rand(1)

    def forward(self, image: torch.Tensor, t: int):
        # t = 0 -> fully pixelated
        # t = T - 1 -> no pixelation
        t = self.T - 1 - t
        fromIndex = t // (self.n_between + 1)
        interpolation = (t % (self.n_between + 1)) / (self.n_between + 1)
        fromImage = self.transforms[fromIndex](image)
        if interpolation == 0:
            return fromImage
        else:
            toIndex = fromIndex + 1
            toImage = self.transforms[toIndex](image)
            return (1 - interpolation) * fromImage + interpolation * toImage


class DDPMCold(nn.Module):
    def __init__(
        self,
        eps_model: nn.Module,
        degradation: Pixelate,
        n_T: int,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super(DDPMCold, self).__init__()
        self.degradation = degradation
        self.eps_model = eps_model
        self.n_T = n_T
        self.criterion = criterion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ts = torch.randint(0, self.n_T, (x.shape[0],)).to(x.device)
        x_t = torch.cat([self.degradation(x, t) for t in _ts], dim=0)
        return self.criterion(x_t, self.eps_model(x_t, _ts / self.n_T))

    def sample(self, device) -> torch.Tensor:
        x_t = self.degradation.get_xT().to(device)
        for s in range(self.n_T - 1, -1, -1):
            x_0 = self.eps_model(x_t, torch.tensor(s / self.n_T).to(device))
            x_t = x_t - self.degradation(x_0, s) + self.degradation(x_0, s - 1)
        return x_t


class DDPM(nn.Module):
    def __init__(
        self,
        eps_model: nn.Module,
        betas: Tuple[float, float],
        n_T: int,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super(DDPM, self).__init__()
        self.eps_model = eps_model

        # register_buffer allows us to freely access these tensors by name. It helps device placement.
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.criterion = criterion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Makes forward diffusion x_t, and tries to guess epsilon value from x_t using eps_model.
        This implements Algorithm 1 in the paper.
        """

        _ts = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(x.device)
        # t ~ Uniform(0, n_T)
        eps = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * eps
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        return self.criterion(eps, self.eps_model(x_t, _ts / self.n_T))

    def sample(self, n_sample: int, size, device) -> torch.Tensor:
        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)

        # This samples accordingly to Algorithm 2. It is exactly the same logic.
        for i in range(self.n_T, 0, -1):
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            eps = self.eps_model(
                x_i, torch.tensor(i / self.n_T).to(device).repeat(n_sample, 1)
            )
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )

        return x_i


def ddpm_schedules(beta1: float, beta2: float, T: int) -> Dict[str, torch.Tensor]:
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
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
    }
