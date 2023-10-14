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
import wandb


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


class TimeEmbed(nn.Module):
    """https://distill.pub/2018/feature-wise-transformations/"""

    def __init__(self, input_size, output_size):
        super(TimeEmbed, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.GELU(),
            nn.Linear(output_size, output_size),
        )

    def forward(
        self, channel_block: torch.Tensor, time_steps: torch.Tensor
    ) -> torch.Tensor:
        embed = self.model(time_steps)
        embed = embed.view(embed.shape[0], embed.shape[1], 1, 1)
        embed = channel_block + embed
        return embed


class Block(nn.Module):
    # takes input size and output size
    def __init__(self, in_ch: int, out_ch: int, up_or_down: str):
        super(Block, self).__init__()
        in_ch = int(in_ch)
        out_ch = int(out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.timeEmbed = TimeEmbed(1, in_ch)
        self.up_or_down = up_or_down
        if self.up_or_down == "down":
            self.pool = nn.MaxPool2d(2, 2)
        elif self.up_or_down == "up":
            self.upsample = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )
        self.gelu = nn.GELU()
        self.batchnorm = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor, time_steps: torch.Tensor = None) -> torch.Tensor:
        if self.up_or_down == "down":
            x = self.pool(x)

        x = self.timeEmbed(x, time_steps)
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        x = self.gelu(x)
        x = self.batchnorm(x)

        if self.up_or_down == "up":
            x = self.upsample(x)

        return x


class UNet(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super(UNet, self).__init__()
        n = 64
        self.input1 = Block(input_channels, n, "same")
        self.encoder2 = Block(n, 2 * n, "down")
        self.encoder3 = Block(2 * n, 4 * n, "down")
        self.bottle1 = Block(4 * n, 4 * n, "same")
        self.decoder1 = Block(4 * n, 2 * n, "up")
        self.decoder2 = Block(2 * 2 * n, n, "up")  # skip connection -> *2
        self.output1 = Block(2 * n, n, "same")  # skip connection -> *2
        self.final = nn.Conv2d(n, output_channels, kernel_size=3, padding=1)

    def forward(self, x, t) -> torch.Tensor:
        i1 = self.input1(x, t)
        e2 = self.encoder2(i1, t)
        e3 = self.encoder3(e2, t)
        b1 = self.bottle1(e3, t)
        d1 = self.decoder1(b1, t)
        d2 = self.decoder2(torch.cat([d1, e2], dim=1), t)
        o1 = self.output1(torch.cat([d2, i1], dim=1), t)
        f = self.final(o1)
        return f


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

        return self.criterion(eps, self.eps_model(x_t, _ts.unsqueeze(1) / self.n_T))

    def sample(self, n_sample: int, size, device) -> torch.Tensor:
        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)

        # This samples accordingly to Algorithm 2. It is exactly the same logic.
        for i in range(self.n_T, 0, -1):
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            time = torch.tensor([[i / self.n_T]]).to(device)
            eps = self.eps_model(x_i, time)
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )

        return x_i


def train_mnist(n_epoch: int = 100, device="cuda:0") -> None:
    wandb.login()
    wandb.init(project="atia-project", config={}, tags=["mnist"])
    ddpm = DDPM(
        eps_model=UNet(input_channels=1, output_channels=1),
        betas=(1e-4, 0.02),
        n_T=1000,
    )
    ddpm.to(device)

    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))]
    )

    dataset = MNIST(
        "./data",
        train=True,
        download=True,
        transform=tf,
    )
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=20)
    optim = torch.optim.Adam(ddpm.parameters(), lr=2e-3)

    for i in range(n_epoch):
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
            model_path = f"ddpm_mnist.pth"
            save_image(grid, image_path)
            torch.save(ddpm.state_dict(), model_path)
            wandb.log(
                {
                    "epoch": i + 1,
                    "loss": avg_loss,
                    f"sample": wandb.Image(image_path),
                }
            )
            wandb.save(model_path)


if __name__ == "__main__":
    train_mnist()
