import torch
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data


# define our U-net model
class UNet(nn.Module):
    # images have 1 channel (grayscale).

    def __init__(self, stages: int, ctx_sz: int = 1):
        """`stages` defines the number of downsampling and upsampling stages. 0 stages means no downsampling or upsampling."""
        super(UNet, self).__init__()
        self.stages = stages

        c_mult = 16

        self.context_size = ctx_sz
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        self.encoders.append(EncoderBlock(1, c_mult, ctx_sz, False))
        self.encoders.extend(
            [
                EncoderBlock(c_mult * (2 ** (i)), c_mult * (2 ** (i + 1)), ctx_sz)
                for i in range(stages)
            ]
        )

        if stages == 0:
            self.decoders.append(DecoderBlock(c_mult, c_mult, ctx_sz, False))
        else:
            self.decoders.append(
                DecoderBlock(
                    c_mult * 2 ** (stages), c_mult * 2 ** (stages - 1), ctx_sz, True
                )
            )
            self.decoders.extend(
                [
                    DecoderBlock(
                        2 * c_mult * (2 ** (i)), c_mult * (2 ** (i - 1)), ctx_sz
                    )
                    for i in range(stages - 1, 0, -1)
                ]
            )
            self.decoders.append(DecoderBlock(2 * c_mult, c_mult * 1, ctx_sz, False))

        self.final_conv = nn.Conv2d(c_mult, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        # define the forward pass using skip connections

        # encoder
        intermediate_encodings = []
        for i in range(self.stages + 1):
            x = self.encoders[i](x, context)
            intermediate_encodings.append(x)
        intermediate_encodings.pop()  # we don't need to concatenate the last layer as it goes directly to the decoder

        intermediate_encodings.reverse()

        # decoder
        for i in range(self.stages + 1):
            if i > 0:
                # concatenate the previous conv in the encoding stage to feed to the decoding (skip connection)
                x = torch.cat((x, intermediate_encodings[i - 1]), dim=1)
            x = self.decoders[i](x)

        x = self.final_conv(x)

        # exit()
        return x


class EncoderBlock(nn.Module):
    # takes input size and output size
    def __init__(self, in_ch: int, out_ch: int, ctx_sz: int, d_smpl: bool = True):
        super(EncoderBlock, self).__init__()
        in_ch = int(in_ch)
        out_ch = int(out_ch)
        self.downsample = d_smpl
        self.context_size = ctx_sz

        # define the layers of the encoder block
        if ctx_sz > 0:
            self.FiLM = FiLM(in_ch, ctx_sz)

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        if d_smpl:
            self.pool = nn.MaxPool2d(2, 2)
        self.gelu = nn.GELU()
        self.batchnorm = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        if self.downsample:
            x = self.pool(x)

        if self.context_size > 0:
            x = self.FiLM(x, context)

        x = self.conv1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        x = self.gelu(x)
        x = self.batchnorm(x)
        return x


class DecoderBlock(nn.Module):
    # takes input size and output size
    def __init__(self, in_ch: int, out_ch: int, ctx_sz: int, up_smpl: bool = True):
        super(DecoderBlock, self).__init__()
        in_ch = int(in_ch)
        out_ch = int(out_ch)
        self.upsample = up_smpl

        # log input params:

        # define the layers of the decoder block
        if up_smpl:
            self.upsample = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.gelu = nn.GELU()
        self.batchnorm = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        x = self.gelu(x)
        x = self.batchnorm(x)
        if self.upsample:
            x = self.upsample(x)
        return x


class FiLM(nn.Module):
    """https://distill.pub/2018/feature-wise-transformations/"""

    def __init__(self, in_ch: int = 256, ctx_size: int = 1):
        super(FiLM, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(ctx_size, in_ch), nn.GELU(), nn.Linear(in_ch, in_ch)
        )

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        embed = self.model(ctx)
        # apply channel-wise affine transformation
        embed = embed.view(embed.shape[0], embed.shape[1], 1, 1)
        embed = x + embed
        return embed
