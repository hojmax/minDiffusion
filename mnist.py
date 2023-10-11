from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from mindiffusion.unet import UNet
from mindiffusion.ddpm import DDPM
from mindiffusion.pyramidal import DDPM as PyramidalDDPM
import matplotlib.pyplot as plt
import wandb
import os
import argparse


def get_ddpm(config: dict) -> nn.Module:
    if config["pyramidal"]:
        # 1 (image) + 2 channels (x,y) * 2 (sin, cos) * degree
        # input_channels = 1 + 2 * 2 * config["positional_degree"]
        input_channels = 1
        unet = UNet(input_channels, config["unet_stages"], config["channel_multiplier"])
        return PyramidalDDPM(
            unet,
            config["betas"],
            config["T_full"],
            config["T_scaled"],
            config["T_delta"],
            config["image_sizes"],
            config["positional_degree"],
            config["batch_size"],
        )
    else:
        input_channels = 1
        unet = UNet(input_channels, config["unet_stages"], config["channel_multiplier"])
        return DDPM(unet, config["betas"], config["T_full"])


def get_mnist(image_size: int) -> MNIST:
    tf = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ]
    )
    dataset = MNIST(
        "./data",
        train=True,
        download=True,
        transform=tf,
    )
    return dataset


def train_mnist(config, log_wandb: bool) -> None:
    if log_wandb:
        wandb.login()
        wandb.init(project="atia-project", config=config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ddpm = get_ddpm(config)
    ddpm.to(device)

    os.makedirs("images", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    dataset = get_mnist(max(config["image_sizes"]))

    if config["only_0_1"]:
        dataset.data = dataset.data[dataset.targets <= 1]
        dataset.targets = dataset.targets[dataset.targets <= 1]

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )
    optim = torch.optim.Adam(ddpm.parameters(), lr=config["lr"])
    for i in range(config["n_epoch"]):
        ddpm.train()
        total_loss = 0

        progess_bar = tqdm(dataloader)
        loss_avg = None
        first_in_epoch = True
        for x, _ in progess_bar:
            optim.zero_grad()
            x = x.to(device)
            loss = ddpm(x, first_in_epoch)
            first_in_epoch = False
            loss.backward()
            total_loss += loss.item()
            optim.step()

            if loss_avg is None:
                loss_avg = loss.item()
            else:
                loss_avg = 0.9 * loss_avg + 0.1 * loss.item()

            progess_bar.set_description(f"loss: {loss_avg:.4f}")

        avg_loss = total_loss / len(dataloader)
        # ddpm.eval()
        # sample_save_path = f"images/ddpm_sample_{i}.png"
        # forward_save_path = f"images/ddpm_forward_{i}.png"
        # model_save_path = f"models/ddpm_mnist_{i}.pth"

        # with torch.no_grad():
        #     n_images = 4
        #     # image_size = (1, max(config["image_sizes"]), max(config["image_sizes"]))
        #     xh = ddpm.sample(
        #         n_images**2,
        #         # image_size,
        #         device,
        #         # i,
        #     )
        #     real_images, _ = next(iter(dataloader))
        #     real_images = real_images[: n_images**2].to(device)
        #     real_grid = make_grid(real_images, nrow=n_images).to(device)
        #     fake_grid = make_grid(xh, nrow=n_images).to(device)
        #     result_grid = torch.cat((fake_grid, real_grid), dim=-1)
        #     save_image(result_grid, sample_save_path)
        #     torch.save(ddpm.state_dict(), model_save_path)

        if log_wandb:
            wandb.log(
                {
                    "epoch": i + 1,
                    "loss": avg_loss,
                    # f"sample": wandb.Image(sample_save_path),
                    # f"forward": wandb.Image(forward_save_path),
                }
            )
            # wandb.save(model_save_path)


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
        "image_sizes": [8],
        "channel_multiplier": 16,
        "only_0_1": True,
        "pyramidal": False,
        "T_full": 100,
        "T_scaled": 100,
        "T_delta": 0.5,
        "positional_degree": 1,
    }
    train_mnist(config, args.wandb)
