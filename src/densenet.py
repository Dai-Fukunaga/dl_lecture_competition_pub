import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers, p_drop=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                BottleneckLayer(in_channels + i * growth_rate, growth_rate, p_drop)
            )

    def forward(self, x):
        for layer in self.layers:
            new_features = layer(x)
            x = torch.cat([x, new_features], dim=1)
        return x


class BottleneckLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, p_drop):
        super().__init__()
        inter_channels = 4 * growth_rate
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels, inter_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(inter_channels)
        self.conv2 = nn.Conv1d(inter_channels, growth_rate, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x):
        out = self.conv1(F.gelu(self.bn1(x)))
        out = self.conv2(F.gelu(self.bn2(out)))
        return self.dropout(out)


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_channels)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(F.gelu(self.bn(x)))
        return self.pool(x)


class DenseNetClassifier(nn.Module):
    def __init__(
        self,
        num_classes,
        seq_len,
        in_channels,
        growth_rate=32,
        num_init_features=64,
        num_blocks=3,
        num_layers_per_block=4,
        p_drop=0.1,
    ):
        super().__init__()
        self.conv0 = nn.Conv1d(
            in_channels, num_init_features, kernel_size=7, stride=2, padding=3
        )
        self.bn0 = nn.BatchNorm1d(num_init_features)
        self.pool0 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        num_features = num_init_features
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(
                DenseBlock(num_features, growth_rate, num_layers_per_block, p_drop)
            )
            num_features = num_features + num_layers_per_block * growth_rate
            if i != num_blocks - 1:
                self.blocks.append(TransitionLayer(num_features, num_features // 2))
                num_features = num_features // 2

        self.head = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(num_features, num_classes),
        )

    def forward(self, x):
        x = self.pool0(F.gelu(self.bn0(self.conv0(x))))
        for block in self.blocks:
            x = block(x)
        return self.head(x)
