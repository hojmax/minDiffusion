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
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
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


class TimeSiren(nn.Module):
    def __init__(self, emb_dim: int) -> None:
        super(TimeSiren, self).__init__()

        self.lin1 = nn.Linear(1, emb_dim, bias=False)
        self.lin2 = nn.Linear(emb_dim, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 1)
        x = torch.sin(self.lin1(x))
        x = self.lin2(x)
        return x


class NaiveUnet(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, u_channels: int = 256
    ) -> None:
        super(NaiveUnet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.base_channels = u_channels

        self.init_conv = Conv3(in_channels, u_channels, is_res=True)

        self.down1 = UnetDown(u_channels, u_channels)
        self.down2 = UnetDown(u_channels, 2 * u_channels)

        self.to_vec = nn.Sequential(nn.AvgPool2d(4), nn.ReLU())

        self.timeembed = TimeSiren(2 * u_channels)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * u_channels, 2 * u_channels, 7, 7),
            nn.GroupNorm(8, 2 * u_channels),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * u_channels, u_channels)
        self.up2 = UnetUp(2 * u_channels, u_channels)
        self.out = nn.Conv2d(2 * u_channels, self.out_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)
        print("x", x.shape)
        down1 = self.down1(x)
        print("down1", down1.shape)
        down2 = self.down2(down1)
        print("down2", down2.shape)

        thro = self.to_vec(down2)
        print("thro1", thro.shape)
        temb = self.timeembed(t).view(-1, self.base_channels * 2, 1, 1)
        print("temb", temb.shape)

        thro = self.up0(thro + temb)
        print("thro2", thro.shape)

        print("inner", self.up1(thro, down2).shape)
        up1 = self.up1(thro, down2) + temb
        print("up1", up1.shape)
        up2 = self.up2(up1, down1)
        print("up2", up2.shape)

        out = self.out(torch.cat((up2, x), 1))
        print("out", out.shape)

        return out
