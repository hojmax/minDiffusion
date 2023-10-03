from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from mindiffusion.unet import UNet
from mindiffusion.ddpm import DDPM
import wandb
import os


def get_mnist_dataset():
    resize_32_tensor = transforms.Compose(
        [transforms.Resize(32), transforms.ToTensor()]
    )
    dataset = MNIST(
        "./data",
        train=True,
        download=True,
        transform=resize_32_tensor,
    )
    return dataset


def save_sample(ddpm, path):
    ddpm.eval()
    with torch.no_grad():
        xh = ddpm.sample(16, (1, 32, 32), ddpm.device)
        grid = make_grid(xh, nrow=4)
        save_image(grid, path)


def save_model(ddpm, path):
    torch.save(ddpm.state_dict(), path)
    wandb.save(path)


def train_mnist() -> None:
    config = {
        "n_epoch": 100,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "batch_size": 64,
        "lr": 2e-4,
        "betas": (1e-4, 0.02),
        "n_T": 1000,
        "unet_stages": 3,
    }
    wandb.login()
    wandb.init(project="atia-project", config=config)

    ddpm = DDPM(UNet(config["unet_stages"]), config["betas"], config["n_T"])
    ddpm.to(config["device"])

    os.makedirs("images", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    dataloader = DataLoader(
        get_mnist_dataset(),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=2,
    )
    optim = torch.optim.Adam(ddpm.parameters(), lr=config["lr"])
    for i in range(config["n_epoch"]):
        ddpm.train()
        total_loss = 0

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, _ in pbar:
            optim.zero_grad()
            x = x.to(ddpm.device)
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

        image_save_path = f"images/ddpm_sample_{i}.png"
        model_save_path = f"models/ddpm_mnist_{i}.pth"
        save_sample(ddpm, image_save_path)
        save_model(ddpm, model_save_path)

        wandb.log(
            {
                "epoch": i + 1,
                "loss": avg_loss,
                f"sample": wandb.Image(image_save_path),
            }
        )


if __name__ == "__main__":
    train_mnist()
