import torch
import torch.nn as nn
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from squeezeformer.encoder import SqueezeformerBlock


class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()

        hidden_size = input_size + output_size
        hidden_sizes = [
            3 * hidden_size,
            2 * hidden_size,
            hidden_size,
            2 * hidden_size,
            3 * hidden_size,
        ]

        # Initialize the layers
        layers = []
        previous_size = input_size  # input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(previous_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))  # Normalization layer
            layers.append(nn.LeakyReLU(inplace=True))  # Activation
            layers.append(nn.Dropout(p=0.1))  # Dropout for regularization
            previous_size = hidden_size

        # Output layer - no dropout, no activation function
        layers.append(nn.Linear(previous_size, output_size))

        # Register all layers
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class Swish(nn.Module):
    def __init__(self) -> None:
        super(Swish, self).__init__()

    def forward(self, inputs):
        return inputs * inputs.sigmoid()


class DoubleLevelWiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, level_size=3):
        super(DoubleLevelWiseConv, self).__init__()
        # input: (batch_size, 25, 60, 1)

        self.bn_00 = nn.BatchNorm2d(in_channels)
        self.depthwise_01 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(level_size, 1),
            stride=1,
            padding=((level_size - 1) // 2, 0),
        )
        self.silu = Swish()
        self.bn_01 = nn.BatchNorm2d(out_channels)
        self.depthwise_02 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(level_size, 1),
            stride=1,
            padding=((level_size - 1) // 2, 0),
        )
        self.silu = Swish()
        self.bn_02 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn_00(x)
        x = self.depthwise_01(x)
        x = self.silu(x)
        x = self.bn_01(x)
        x = self.depthwise_02(x)
        x = self.silu(x)
        x = self.bn_02(x)
        return x


class ChannelMixing(nn.Module):
    def __init__(self, in_channels, out_channels, level_size=3):
        super(ChannelMixing, self).__init__()
        # input: (batch_size, 60, channel)

        self.first_linear = nn.Linear(in_channels, out_channels)
        self.gelu = Swish()
        self.layer_norm = nn.LayerNorm(out_channels)
        self.second_linear = nn.Linear(out_channels, in_channels)

    def forward(self, x):
        residual = x
        x = self.first_linear(x)
        x = self.gelu(x)
        x = self.layer_norm(x)
        x = self.second_linear(x)
        return x + residual


class LeapNetwork(nn.Module):
    def __init__(self):
        super(LeapNetwork, self).__init__()

        input_channels = 25
        cnn_channels = 64
        duration = 60

        self.level_wise_conv_01 = DoubleLevelWiseConv(
            in_channels=input_channels, out_channels=cnn_channels, level_size=3
        )
        self.level_wise_conv_02 = DoubleLevelWiseConv(
            in_channels=input_channels, out_channels=cnn_channels, level_size=7
        )
        self.level_wise_conv_03 = DoubleLevelWiseConv(
            in_channels=input_channels, out_channels=cnn_channels, level_size=15
        )
        self.channel_mixer_01 = ChannelMixing(
            in_channels=cnn_channels, out_channels=cnn_channels * 2
        )
        self.channel_mixer_02 = ChannelMixing(
            in_channels=cnn_channels, out_channels=cnn_channels * 2
        )
        self.channel_mixer_03 = ChannelMixing(
            in_channels=cnn_channels, out_channels=cnn_channels * 2
        )
        self.level_wise_conv_04 = DoubleLevelWiseConv(
            in_channels=cnn_channels * 3, out_channels=cnn_channels * 3, level_size=3
        )
        self.squeezeformers = nn.ModuleList(
            [SqueezeformerBlock(encoder_dim=cnn_channels * 3) for _ in range(6)]
        )

        self.linear = nn.Linear(cnn_channels * 3, 14 - 8)
        self.mlp = MLP(556, 8)

    def forward(self, g0, g1, g2, g3, g4, g5, g6, g7, g8, g_else):
        flatten_feature = torch.cat(
            [g0, g1, g2, g3, g4, g5, g6, g7, g8, g_else], dim=1
        )  # (batch_size, 556)

        g0 = g0.unsqueeze(1)
        g1 = g1.unsqueeze(1)
        g2 = g2.unsqueeze(1)
        g3 = g3.unsqueeze(1)
        g4 = g4.unsqueeze(1)
        g5 = g5.unsqueeze(1)
        g6 = g6.unsqueeze(1)
        g7 = g7.unsqueeze(1)
        g8 = g8.unsqueeze(1)

        g_else = g_else.unsqueeze(-1)  # (batch_size, 16-channel, 1)
        g_else = g_else.expand(-1, -1, g6.size(-1))  # (batch_size, 16-channel, 60)

        # input: (batch_size, 7, 60) => output: (batch_size, 14, 60)
        x = torch.cat(
            [g0, g1, g2, g3, g4, g5, g6, g7, g8, g_else], dim=1
        )  # torch.Size([100, 25, 60])

        x = x.unsqueeze(1)
        # print(x.shape)  # 100, 1, 25, 60
        x = x.permute(0, 2, 3, 1)
        # print(x.shape)  # 100, 25, 60, 1

        level_conved_01 = self.level_wise_conv_01(x).squeeze(3)  # 100, 64, 60
        level_conved_02 = self.level_wise_conv_02(x).squeeze(3)  # 100, 64, 60
        level_conved_03 = self.level_wise_conv_03(x).squeeze(3)  # 100, 64, 60

        # channel-mixing input: (batch_size, 60, channel)
        level_conved_01 = level_conved_01.permute(0, 2, 1)  # 100, 60, 64
        level_conved_02 = level_conved_02.permute(0, 2, 1)  # 100, 60, 64
        level_conved_03 = level_conved_03.permute(0, 2, 1)  # 100, 60, 64
        mixed_01 = self.channel_mixer_01(level_conved_01)  # 100, 60, 64
        mixed_02 = self.channel_mixer_02(level_conved_02)  # 100, 60, 64
        mixed_03 = self.channel_mixer_03(level_conved_03)  # 100, 60, 64

        x = torch.cat([mixed_01, mixed_02, mixed_03], dim=2)  # 100, 60, 128
        x = x.permute(0, 2, 1)  # 100, 128, 60
        x = x.unsqueeze(-1)  # 100, 128, 60, 1

        # x = self.level_wise_conv_03(x).squeeze(3)  # 100, 128, 60
        x = x.squeeze(3)  # 100, 128, 60

        x = x.permute(0, 2, 1)  # 100, 60, 128

        for i, layer in enumerate(self.squeezeformers):
            x = layer(x)

        x = self.linear(x)  # 100, 60, 14
        x = x.transpose(1, 2)  # 100, 14, 60

        """
        x = self.pointwise(x)
        # print(x.shape)  # 100, 1, 25, 60
        x = self.depthwise(x)
        # print(x.shape)  # 100, 1, 25, 60
        x = x.squeeze(1)  # 100, 25, 60
        x = self.pointwise_upchannel(x)  # 100, 64, 60
        x = x.permute(0, 2, 1)  # 100, 60, 64

        for i, layer in enumerate(self.squeezeformers):
            x = layer(x)

        x = self.linear(x)  # 100, 60, 14
        x = x.transpose(1, 2)  # 100, 14, 60
        """
        mlp_pred = self.mlp(flatten_feature).unsqueeze(-1)  # (batch_size, 8, 1)
        mlp_pred = mlp_pred.expand(-1, -1, 60)  # (batch_size, 8, 60)
        x = torch.cat([x, mlp_pred], dim=1)  # (batch_size, 60, 14)
        return x


class UNet1DDecoder(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        duration: int,
        bilinear: bool = False,
        se: bool = False,
        res: bool = False,
        scale_factor: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.duration = duration
        self.bilinear = bilinear
        self.se = se
        self.res = res
        self.scale_factor = scale_factor

        factor = 2 if bilinear else 1
        self.inc = DoubleConv(
            self.n_channels, 64, norm=partial(create_layer_norm, length=self.duration)
        )
        self.down1 = Down(
            64,
            128,
            scale_factor,
            norm=partial(create_layer_norm, length=self.duration // 2),
        )
        self.down2 = Down(
            128,
            256,
            scale_factor,
            norm=partial(create_layer_norm, length=self.duration // 4),
        )
        self.down3 = Down(
            256,
            512,
            scale_factor,
            norm=partial(create_layer_norm, length=self.duration // 8),
        )
        self.down4 = Down(
            512,
            1024 // factor,
            scale_factor,
            norm=partial(create_layer_norm, length=self.duration // 16),
        )
        self.up1 = Up(
            1024,
            512 // factor,
            bilinear,
            scale_factor,
            norm=partial(create_layer_norm, length=self.duration // 8),
        )
        self.up2 = Up(
            512,
            256 // factor,
            bilinear,
            scale_factor,
            norm=partial(create_layer_norm, length=self.duration // 4),
        )
        self.up3 = Up(
            256,
            128 // factor,
            bilinear,
            scale_factor,
            norm=partial(create_layer_norm, length=self.duration // 2),
        )
        self.up4 = Up(
            128,
            64,
            bilinear,
            scale_factor,
            norm=partial(create_layer_norm, length=self.duration),
        )

        self.cls = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, self.n_classes, kernel_size=1, padding=0),
            nn.Dropout(dropout),
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self, x: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> dict[str, Optional[torch.Tensor]]:
        """Forward

        Args:
            x (torch.Tensor): (batch_size, n_channels, n_timesteps)

        Returns:
            torch.Tensor: (batch_size, n_timesteps, n_classes)
        """
        # 1D U-Net
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # classifier
        logits = self.cls(x)  # (batch_size, n_classes, n_timesteps)
        return logits.transpose(1, 2)  # (batch_size, n_timesteps, n_classes)


class SEModule(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        (
            b,
            c,
            _,
        ) = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels=None,
        norm=nn.BatchNorm1d,
        se=False,
        res=False,
    ):
        super().__init__()
        self.res = res
        if not mid_channels:
            mid_channels = out_channels
        if se:
            non_linearity = SEModule(out_channels)
        else:
            non_linearity = nn.ReLU(inplace=True)
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            norm(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm(out_channels),
            non_linearity,
        )

    def forward(self, x):
        if self.res:
            return x + self.double_conv(x)
        else:
            return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(
        self,
        in_channels,
        out_channels,
        scale_factor,
        norm=nn.BatchNorm1d,
        se=False,
        res=False,
    ):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(scale_factor),
            DoubleConv(in_channels, out_channels, norm=norm, se=se, res=res),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self,
        in_channels,
        out_channels,
        bilinear=False,
        scale_factor=2,
        norm=nn.BatchNorm1d,
    ):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=scale_factor, mode="linear", align_corners=True
            )
            self.conv = DoubleConv(
                in_channels, out_channels, in_channels // 2, norm=norm
            )
        else:
            self.up = nn.ConvTranspose1d(
                in_channels,
                in_channels // 2,
                kernel_size=scale_factor,
                stride=scale_factor,
            )
            self.conv = DoubleConv(in_channels, out_channels, norm=norm)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diff = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, [diff // 2, diff - diff // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


def create_layer_norm(channel, length):
    return nn.LayerNorm([channel, length])
