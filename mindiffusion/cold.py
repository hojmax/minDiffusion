import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import random


class Pixelate:
    def __init__(self, sizes: list[int], n_between: int = 1):
        """Sizes is a list of ints from smallest to largest"""
        self.sizes = sizes
        self.transforms = [self.set_image_to_random_grey]
        interpolation = transforms.InterpolationMode.NEAREST
        for size in sizes:
            self.transforms.append(
                transforms.Compose(
                    [
                        transforms.Resize(size, interpolation),
                        transforms.Resize(sizes[-1], interpolation),
                    ]
                )
            )
        self.n_between = n_between
        self.T = Pixelate.calculate_T(len(sizes), n_between)

    def calculate_T(n_sizes: int, n_between: int):
        """
        img0 -> img1/N -> img2/N -> .. -> img(N-1)/N -> img1 -> img(N+1)/N ->... imgK
        Where a fractional image denotes a interpolation between two images (imgA and img(A+1))
        The number of images in the above becomes (excluding the original image):
        K * (N+1)
        """
        return n_sizes * (n_between + 1)

    def seed(self, seed: int):
        torch.manual_seed(seed)

    def get_xT(self):
        """Returns a random grey image of max size"""
        size = self.sizes[-1]
        color = torch.rand(1)
        return torch.ones(1, size, size) * color

    def set_image_to_random_grey(self, image: torch.Tensor):
        return image * 0 + torch.rand(1)

    def __call__(self, image: torch.Tensor, t: int):
        """
        t = 0 -> no pixelation
        t = T -> full pixelations
        """
        fromIndex = (self.T - t) // (self.n_between + 1)
        interpolation = ((self.T - t) % (self.n_between + 1)) / (self.n_between + 1)
        fromImage = self.transforms[fromIndex](image)
        if interpolation == 0:
            return fromImage
        else:
            toIndex = fromIndex + 1
            toImage = self.transforms[toIndex](image)
            return (1 - interpolation) * fromImage + interpolation * toImage


class DDPMCold(nn.Module):
    def __init__(
        self,
        eps_model: nn.Module,
        degradation: Pixelate,
        n_T: int,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super(DDPMCold, self).__init__()
        self.degradation = degradation
        self.eps_model = eps_model
        self.n_T = n_T
        self.criterion = criterion

    def forward(self, x: torch.Tensor, should_show: bool) -> torch.Tensor:
        _ts = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(x.device)
        x_t = torch.cat(
            [self.degradation(x[i], _ts[i]).unsqueeze(0) for i in range(x.shape[0])]
        )
        model_output = self.eps_model(x_t, _ts.unsqueeze(1) / self.n_T)
        if should_show:
            fig = plt.figure(figsize=(20, 5))
            n_images = 10
            for i in range(n_images):
                image = x_t[i]
                ax = fig.add_subplot(3, n_images, i + 1)
                ax.axis("off")
                ax.set_title("t={}".format(_ts[i]))
                ax.imshow(image.squeeze(), cmap="gray", vmin=0, vmax=1)
                ax = fig.add_subplot(3, n_images, i + 1 + n_images)
                ax.axis("off")
                ax.imshow(x[i].squeeze(), cmap="gray", vmin=0, vmax=1)
                ax = fig.add_subplot(3, n_images, i + 1 + 2 * n_images)
                ax.axis("off")
                ax.imshow(
                    model_output[i].detach().squeeze(), cmap="gray", vmin=0, vmax=1
                )
            plt.show()
        return self.criterion(x, model_output)

    def sample(self, _n_sample, _size, device, epoch) -> torch.Tensor:
        final = []
        all_steps = []
        for i in range(_n_sample):
            random_seed = random.randint(0, 100000)
            steps = []
            self.degradation.seed(random_seed)
            x_t = self.degradation.get_xT().unsqueeze(0).to(device)
            for s in range(self.n_T, 1, -1):
                steps.append(x_t.squeeze(0))
                x_0 = self.eps_model(x_t, torch.tensor([[s / self.n_T]]).to(device))
                self.degradation.seed(random_seed)
                d_s = self.degradation(x_0, s)
                self.degradation.seed(random_seed)
                d_s1 = self.degradation(x_0, s - 1)
                x_t = x_t - d_s + d_s1
            steps.append(x_t.squeeze(0))
            all_steps.append(torch.cat(steps, 1))
            final.append(x_t.squeeze(0))
        save_image(
            torch.cat(all_steps, 2),
            f"images/cold_{epoch}.png",
        )

        return final


if __name__ == "__main__":

    def showImageList(images):
        # show images side by side
        fig = plt.figure(figsize=(20, 5))
        for i, image in enumerate(images):
            ax = fig.add_subplot(1, len(images), i + 1)
            # show i above image
            ax.set_title("t={}".format(i))
            ax.axis("off")
            ax.imshow(image.squeeze(), cmap="gray", vmin=0, vmax=1)
        plt.show()

    tf = transforms.Compose(
        [
            transforms.Resize(16),
            transforms.ToTensor(),
        ]
    )
    dataset = MNIST(
        "./data",
        train=True,
        download=True,
        transform=tf,
    )
    degradation = Pixelate([4, 8, 16], 5)
    print(degradation.T)
    print(Pixelate.calculate_T(3, 5))
    test_image = dataset[0][0]
    showImageList(
        [
            degradation.seed(17) or degradation(test_image, t)
            for t in range(degradation.T + 1)
        ]
    )
