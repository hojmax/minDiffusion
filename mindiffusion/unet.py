"""
Simple Unet Structure.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(),
            nn.GELU(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(),
            nn.GELU(),
        )

        self.is_res = is_res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.main(x)
        if self.is_res:
            x = x + self.conv(x)
            return x / 1.414
        else:
            return self.conv(x)


class UnetDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(UnetDown, self).__init__()
        layers = [Conv3(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(UnetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            Conv3(out_channels, out_channels),
            Conv3(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x, skip), 1)
        x = self.model(x)

        return x


class Embedding(nn.Module):
    def __init__(self, emb_dim: int) -> None:
        super(Embedding, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(1, emb_dim), nn.GELU(), nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 1)
        return self.model(x)


class NaiveUnet(nn.Module):
    def __init__(self, in_channels: int, n_feat: int = 256) -> None:
        super(NaiveUnet, self).__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat

        self.init_conv = Conv3(in_channels, n_feat, is_res=True)
        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.timeembed = Embedding(2 * n_feat)

        self.mid_block = nn.Sequential(
            nn.Conv2d(2 * n_feat, 2 * n_feat, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(2 * n_feat, 2 * n_feat, kernel_size=3, padding=1),
            nn.GELU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, 3, 1, 1),
            nn.BatchNorm2d(),
            nn.GELU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)
        # print("init", x.shape)
        down1 = self.down1(x)
        # print("down1", down1.shape)
        down2 = self.down2(down1)
        # print("down2", down2.shape)
        mid = self.mid_block(down2)
        # print("mid", mid.shape)
        time_embed = self.timeembed(t)
        # print("time_embed", time_embed.shape)
        mid = mid + time_embed.view(-1, 2 * self.n_feat, 1, 1)
        # print("mid2", mid.shape)
        up1 = self.up1(mid, down2)
        # print("up1", up1.shape)
        up2 = self.up2(up1, down1)
        # print("up2", up2.shape)
        out = self.out(up2)
        # print("out", out.shape)
        return out
