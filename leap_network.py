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
