"""
Extremely Minimalistic Implementation of DDPM

https://arxiv.org/abs/2006.11239

Everything is self contained. (Except for pytorch and torchvision... of course)

run it with `python superminddpm.py`
"""

from typing import Dict, Tuple
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from mindiffusion.unet import NaiveUnet
import matplotlib.pyplot as plt
import wandb
import os


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

        _ts = torch.randint(1, self.n_T, (x.shape[0],)).to(
            x.device
        )  # t ~ Uniform(0, n_T)
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
            eps = self.eps_model(x_i, torch.tensor([i / self.n_T]).to(device))
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )

        return x_i


def train_mnist(n_epoch: int = 100, device="cuda:0") -> None:
    wandb.login()
    wandb.init(project="atia-project", config={}, tags=["mnist"])
    ddpm = DDPM(eps_model=NaiveUnet(1, n_feat=128), betas=(1e-4, 0.02), n_T=100)
    ddpm.to(device)

    os.makedirs("images", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    tf = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (1.0)),
        ]
    )

    dataset = MNIST(
        "./data",
        train=True,
        download=True,
        transform=tf,
    )
    # print(dataset[0][0].shape)
    # plt.imshow(dataset[0][0].squeeze().numpy())
    # plt.show()

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8)

    optim = torch.optim.Adam(ddpm.parameters(), lr=2e-4)
    xh = ddpm.sample(1, (1, 28, 28), device)
    global_steps = 0

    for i in range(n_epoch):
        ddpm.train()
        total_loss = 0

        pbar = tqdm(dataloader)
        for x, _ in pbar:
            optim.zero_grad()
            x = x.to(device)
            loss = ddpm(x)
            loss.backward()
            total_loss += loss.item()
            global_steps += 1
            optim.step()

        avg_loss = total_loss / len(dataloader)
        ddpm.eval()
        with torch.no_grad():
            xh = ddpm.sample(4, (1, 28, 28), device)
            grid = make_grid(xh, nrow=2)
            save_image(grid, f"images/ddpm_sample_{i}.png")

            # save model
            torch.save(ddpm.state_dict(), f"models/ddpm_mnist.pth")

        wandb.log(
            {
                "epoch": i + 1,
                "loss": avg_loss,
                "global_steps": global_steps,
                f"sample": wandb.Image(f"images/ddpm_sample_{i}.png"),
            }
        )


if __name__ == "__main__":
    train_mnist()
