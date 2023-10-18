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
from model import Model
import wandb
import argparse


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

        time = torch.randint(1, self.n_T, (x.shape[0],)).to(
            x.device
        )  # t ~ Uniform(0, n_T)
        eps = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[time, None, None, None] * x
            + self.sqrtmab[time, None, None, None] * eps
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        return self.criterion(eps, self.eps_model(x_t, time))

    def sample(self, n_sample: int, size, device) -> torch.Tensor:
        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)

        # This samples accordingly to Algorithm 2. It is exactly the same logic.
        for i in range(self.n_T, 0, -1):
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            time = (torch.ones(n_sample) * i).to(device)
            eps = self.eps_model(x_i, time)
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )

        x_i = torch.clamp(x_i, 0, 1)
        return x_i


def train_mnist(log_wandb) -> None:
    model_config = {
        "resolution": 28,
        "in_channels": 1,
        "out_ch": 1,
        "ch": 128,
        "ch_mult": (
            1,
            2,
        ),
        "num_res_blocks": 3,
        "attn_resolutions": (14,),
        "dropout": 0.1,
    }
    config = {
        **model_config,
        "betas": (1e-4, 0.02),
        "n_T": 1000,
        "n_epoch": 100,
        "lr": 2e-3,
        "batch_size": 128,
    }
    if log_wandb:
        wandb.login()
        wandb.init(project="atia-project", config=config)
    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    ddpm = DDPM(
        eps_model=Model(**model_config),
        betas=config["betas"],
        n_T=config["n_T"],
    )
    ddpm.to(device)

    tf = transforms.ToTensor()

    dataset = MNIST(
        "./data",
        train=True,
        download=True,
        transform=tf,
    )
    dataloader = DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=True, num_workers=2
    )
    optim = torch.optim.Adam(ddpm.parameters(), lr=config["lr"])

    for i in range(config["n_epoch"]):
        ddpm.train()

        pbar = tqdm(dataloader)
        loss_ema = None
        total_loss = 0
        for x, _ in pbar:
            optim.zero_grad()
            x = x.to(device)
            loss = ddpm(x)
            loss.backward()
            total_loss += loss.item()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()
        avg_loss = total_loss / len(dataloader)
        ddpm.eval()
        with torch.no_grad():
            xh = ddpm.sample(16, (1, 28, 28), device)
            grid = make_grid(xh, nrow=4)
            image_path = f"ddpm_sample_{i}.png"
            model_path = f"ddpm_mnist_{i}.pth"
            save_image(grid, image_path)
            torch.save(ddpm.state_dict(), model_path)
            if log_wandb:
                wandb.log(
                    {
                        "epoch": i + 1,
                        "loss": avg_loss,
                        f"sample": wandb.Image(image_path),
                    }
                )
                wandb.save(model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_wandb", action="store_true")
    args = parser.parse_args()
    train_mnist(args.log_wandb)
