from tqdm import tqdm
from mindiffusion.cold import DDPMCold, Pixelate
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from mindiffusion.unet import UNet
from mindiffusion.ddpm import DDPM
import matplotlib.pyplot as plt
import wandb
import os
import argparse


def get_ddpm(config: dict) -> nn.Module:
    unet = UNet(config["unet_stages"], config["c_mult"])

    if config["cold"]:
        degradation = Pixelate(config["cold"]["sizes"], config["cold"]["n_between"])
        return DDPMCold(unet, degradation, config["n_T"])
    else:
        return DDPM(unet, config["betas"], config["n_T"])


def train_mnist(config, log_wandb: bool) -> None:
    if log_wandb:
        wandb.login()
        wandb.init(project="atia-project", config=config)

    ddpm = get_ddpm(config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ddpm.to(device)

    os.makedirs("images", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    tf = transforms.Compose(
        [
            transforms.Resize(config["image_size"]),
            transforms.ToTensor(),
        ]
    )

    dataset = MNIST(
        "./data",
        train=True,
        download=True,
        transform=tf,
    )

    if config["only_0_1"]:
        dataset.data = dataset.data[dataset.targets <= 1]

    dataloader = DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=True, num_workers=2
    )
    optim = torch.optim.Adam(ddpm.parameters(), lr=config["lr"])
    for i in range(config["n_epoch"]):
        ddpm.train()
        total_loss = 0

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, _ in pbar:
            optim.zero_grad()
            x = x.to(device)
            loss = ddpm(x)
            loss.backward()
            total_loss += loss.item()
            optim.step()

            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()

            pbar.set_description(f"loss: {loss_ema:.4f}")

        avg_loss = total_loss / len(dataloader)
        ddpm.eval()
        image_save_path = f"images/ddpm_sample_{i}.png"
        model_save_path = f"models/ddpm_mnist_{i}.pth"

        with torch.no_grad():
            n_images = 4
            xh = ddpm.sample(
                n_images**2, (1, config["image_size"], config["image_size"]), device
            )
            real_images, _ = next(iter(dataloader))
            real_images = real_images[: n_images**2].to(device)
            real_grid = make_grid(real_images, nrow=n_images).to(device)
            fake_grid = make_grid(xh, nrow=n_images).to(device)
            result_grid = torch.cat((fake_grid, real_grid), dim=-1)
            save_image(result_grid, image_save_path)
            torch.save(ddpm.state_dict(), model_save_path)

        if log_wandb:
            wandb.log(
                {
                    "epoch": i + 1,
                    "loss": avg_loss,
                    f"sample": wandb.Image(image_save_path),
                }
            )

            wandb.save(model_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    config = {
        "n_epoch": 100,
        "batch_size": 64,
        "lr": 4e-4,
        "betas": (1e-4, 0.02),
        "unet_stages": 3,
        "image_size": 16,
        "c_mult": 16,
        "only_0_1": True,
        "cold": {
            "sizes": [8, 16],
            "n_between": 2,
        },
    }
    if config["cold"]:
        # img0 -> img1/N -> img2/N -> .. -> img(N-1)/N -> img1 -> img(N+1)/N ->... imgK
        # Where a fractional image denotes a interpolation between two images (imgA and img(A+1))
        # The number of images in the above becomes:
        # K * N + 1
        # This is n_T
        config["n_T"] = (
            len(config["cold"]["sizes"]) * (config["cold"]["n_between"] + 1) + 1
        )
    else:
        config["n_T"] = 1000
    train_mnist(config, args.wandb)
