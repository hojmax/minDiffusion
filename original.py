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
        self.should_save_pngs = False
        self.last_x = None

        # print(f"ab[{len(self.sqrtab)}]", self.sqrtab)
        # print(f"mab[{len(self.sqrtmab)}]", self.sqrtmab)
        # print("t=", self.n_T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Makes forward diffusion x_t, and tries to guess epsilon value from x_t using eps_model.
        This implements Algorithm 1 in the paper.
        """
        time = torch.randint(1, self.n_T, (x.shape[0],)).to(
            x.device
        )  # (zero indexed, so (0, n_T-1) equal to (1, n_T) in the paper)
        eps = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[time, None, None, None] * x
            + self.sqrtmab[time, None, None, None] * eps
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        return self.criterion(eps, self.eps_model(x_t, time))

    def sample(self, n_sample: int, size, device) -> torch.Tensor:
        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)

        reverse = []
        predictions = []

        if self.should_save_pngs:
            reverse.append(x_i)
        # This samples accordingly to Algorithm 2. It is exactly the same logic.
        for i in range(self.n_T - 1, 0, -1):
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            time = (torch.ones(n_sample) * i).to(device)
            eps = self.eps_model(x_i, time)
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            if self.should_save_pngs:
                reverse.append(x_i)
                predictions.append(
                    self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                )

        predictions.append(torch.zeros_like(x_i))

        if self.should_save_pngs:
            reverse_img = torch.cat(
                [make_image_row(x).squeeze(0).squeeze(0) for x in reverse], dim=0
            )
            predictions_img = torch.cat(
                [make_image_row(x).squeeze(0).squeeze(0) for x in predictions], dim=0
            )
            real_x = self.last_x[:n_sample]

            predictions2 = []
            forward = []
            for t in range(self.n_T - 1, 0, -1):
                eps = torch.randn_like(real_x)
                time = (torch.ones(n_sample) * t).long().to(device)
                forward.append(
                    self.sqrtab[time, None, None, None] * real_x
                    + self.sqrtmab[time, None, None, None] * eps
                )
                eps = self.eps_model(forward[-1], time)
                predictions2.append(
                    self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                )

            forward.append(real_x)
            predictions2.append(torch.zeros_like(real_x))

            forward_img = torch.cat(
                [make_image_row(x).squeeze(0).squeeze(0) for x in forward], dim=0
            )
            predictions2_img = torch.cat(
                [make_image_row(x).squeeze(0).squeeze(0) for x in predictions2], dim=0
            )

            combined_image = torch.cat(
                [forward_img, predictions2_img, reverse_img, predictions_img], dim=1
            )

            save_image(combined_image, f"combined.png")
            exit()

        return x_i


# class DummyNetwork(nn.Module):
#     def __init__(self) -> None:
#         super(DummyNetwork, self).__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(1, 32, 3, 1, 1),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, 3, 1, 1),
#             nn.ReLU(),
#             nn.Conv2d(64, 1, 3, 1, 1),
#         )

#     def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
#         return self.net(x)


def make_image_row(x: torch.Tensor):
    """
    Take tensor (B x C x H x W)
    And outputs a tensor (1 x C x H x W*B)
    Where each of the H x W images are placed side by side
    """
    return torch.cat(torch.split(x, 1, dim=0), dim=3)


def train_mnist(args) -> None:
    model_config = {
        "resolution": 28,
        "in_channels": 1,
        "out_ch": 1,
        "ch": 32,
        "ch_mult": (
            1,
            2,
        ),
        "num_res_blocks": 1,
        "attn_resolutions": (14,),
        "dropout": 0.1,
    }
    if args.log_wandb:
        wandb.login()
        wandb.init(project="atia-project", config=model_config)
    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    ddpm = DDPM(
        eps_model=Model(**model_config),
        betas=(1e-4, 0.02),
        n_T=100,
    )
    ddpm.should_save_pngs = True
    if args.pretrained_model_path:
        ddpm.load_state_dict(
            torch.load(args.pretrained_model_path, map_location=device)
        )
    ddpm.to(device)
    n_epoch = 100

    tf = transforms.ToTensor()

    dataset = MNIST(
        "./data",
        train=True,
        download=True,
        transform=tf,
    )
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)
    optim = torch.optim.Adam(ddpm.parameters(), lr=2e-3)

    for i in range(n_epoch):
        ddpm.train()

        pbar = tqdm(dataloader)
        loss_ema = None
        total_loss = 0
        for x, _ in pbar:
            ddpm.last_x = x.to(device)
            continue
            # if total_loss > 0:
            #     continue
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
            xh = ddpm.sample(2, (1, 28, 28), device)
            grid = make_grid(xh, nrow=4)
            image_path = f"ddpm_sample_{i}.png"
            model_path = f"ddpm_mnist_{i}.pth"
            save_image(grid, image_path)
            torch.save(ddpm.state_dict(), model_path)
            if args.log_wandb:
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
    parser.add_argument("--pretrained_model_path", type=str, default="")
    args = parser.parse_args()
    train_mnist(args)
