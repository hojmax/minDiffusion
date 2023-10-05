from typing import Dict, Tuple
from tqdm import tqdm
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


def train_mnist(log_wandb: bool) -> None:
    config = {
        "n_epoch": 100,
        "batch_size": 64,
        "lr": 4e-4,
        "betas": (1e-4, 0.02),
        "n_T": 1000,
        "unet_stages": 3,
        "image_size": 16,
        "c_mult": 16,
        "only_0_1": True,
        "cold": True,
    }
    if log_wandb:
        wandb.login()
        wandb.init(project="atia-project", config=config)
    unet = UNet(config["unet_stages"], config["c_mult"])
    ddpm = DDPM(unet, config["betas"], config["n_T"])
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
            xh = ddpm.sample(
                16, (1, config["image_size"], config["image_size"]), device
            )
            real_images, _ = next(iter(dataloader))
            real_images = real_images[:16].to(device)
            real_grid = make_grid(real_images, nrow=4).to(device)
            fake_grid = make_grid(xh, nrow=4).to(device)
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
    train_mnist(args.wandb)
