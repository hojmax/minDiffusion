import torch
import torch.nn as nn
from torchvision import transforms


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
        self.n_transforms = len(self.transforms)
        self.T = self.n_transforms + (self.n_transforms - 1) * self.n_between

    def seed(self, seed: int):
        torch.manual_seed(seed)

    def get_xT(self):
        """Returns a random grey image of max size"""
        size = self.sizes
        color = torch.rand(1)
        return torch.ones(1, size, size) * color

    def set_image_to_random_grey(self, image: torch.Tensor):
        return image * 0 + torch.rand(1)

    def forward(self, image: torch.Tensor, t: int):
        # t = 0 -> fully pixelated
        # t = T - 1 -> no pixelation
        t = self.T - 1 - t
        fromIndex = t // (self.n_between + 1)
        interpolation = (t % (self.n_between + 1)) / (self.n_between + 1)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ts = torch.randint(0, self.n_T, (x.shape[0],)).to(x.device)
        x_t = torch.cat([self.degradation(x[i], _ts[i]) for i in range(x.shape[0])])
        return self.criterion(x_t, self.eps_model(x_t, _ts / self.n_T))

    def sample(self, device) -> torch.Tensor:
        x_t = self.degradation.get_xT().to(device)
        for s in range(self.n_T - 1, -1, -1):
            x_0 = self.eps_model(x_t, torch.tensor(s / self.n_T).to(device))
            x_t = x_t - self.degradation(x_0, s) + self.degradation(x_0, s - 1)
        return x_t
