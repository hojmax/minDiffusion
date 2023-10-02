from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from mindiffusion.unet import NaiveUnet
from mindiffusion.ddpm import DDPM
import matplotlib.pyplot as plt
import wandb
import os


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
        ]
    )

    dataset = MNIST(
        "./data",
        train=True,
        download=True,
        transform=tf,
    )

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8)

    optim = torch.optim.Adam(ddpm.parameters(), lr=2e-4)
    xh = ddpm.sample(1, (1, 28, 28), device)
    global_steps = 0

    for i in range(n_epoch):
        ddpm.train()
        total_loss = 0
        loss_ema = None

        pbar = tqdm(dataloader)
        for x, _ in pbar:
            optim.zero_grad()
            x = x.to(device)
            loss = ddpm(x)
            loss.backward()
            total_loss += loss.item()
            global_steps += 1
            optim.step()

            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()

            pbar.set_description(f"loss: {loss_ema:.4f}")

        avg_loss = total_loss / len(dataloader)
        ddpm.eval()
        image_save_path = f"images/ddpm_sample_{i}.png"
        model_save_path = f"models/ddpm_mnist_{i}.pth"

        with torch.no_grad():
            xh = ddpm.sample(9, (1, 28, 28), device)
            grid = make_grid(xh, nrow=3)
            save_image(grid, image_save_path)
            torch.save(ddpm.state_dict(), model_save_path)

        wandb.log(
            {
                "epoch": i + 1,
                "loss": avg_loss,
                "global_steps": global_steps,
                f"sample": wandb.Image(image_save_path),
            }
        )

        wandb.save(model_save_path)


if __name__ == "__main__":
    train_mnist()
