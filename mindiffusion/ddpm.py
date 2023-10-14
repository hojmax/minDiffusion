from typing import Dict, Tuple
import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid


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
        self.saved_x = x
        _ts = torch.randint(1, self.n_T + 1, (x.shape[0],))
        # t ~ Uniform(0, n_T)
        eps = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * eps
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        return self.criterion(
            eps, self.eps_model(x_t, _ts.to(x.device).unsqueeze(1) / self.n_T)
        )

    def sample(self, n_sample: int, size, device) -> torch.Tensor:
        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)
        # reverse_process = []
        # This samples accordingly to Algorithm 2. It is exactly the same logic.
        for i in range(self.n_T, 0, -1):
            # reverse_process.append(x_i)
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            eps = self.eps_model(
                x_i, torch.tensor(i / self.n_T).to(device).repeat(n_sample, 1)
            )
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
        # reverse_process.append(x_i)
        # forward_process = []
        # # Choose n_sample from self.saved_x
        # x_t = self.saved_x[torch.randint(0, self.saved_x.shape[0], (n_sample,))]
        # for t in range(self.n_T, 0, -1):
        #     forward_process.append(x_t)
        #     # add noise to self.saved_x
        #     z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
        #     _ts = torch.tensor(t).to(device).repeat(n_sample, 1)
        #     x_t = (
        #         self.sqrtab[_ts, None, None, None] * x_t
        #         + self.sqrtmab[_ts, None, None, None] * z
        #     )
        # forward_process.append(x_t)

        # interweave the forward and reverse process into an image
        # with 2 * n_sample columns and n_T + 1 rows
        # Every other column is reverse process, and every other is forward process.
        # Then save that image
        # img = torch.cat(
        #     [
        #         torch.cat((f, r), dim=-1)
        #         for f, r in zip(forward_process, reverse_process)
        #     ],
        #     dim=0,
        # )
        # save_image(img, "images/reverse_and_forward.png")
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
