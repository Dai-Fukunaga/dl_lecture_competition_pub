import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.3):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.dropout(F.leaky_relu(out))

        return out


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.5):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(
            out_channels, out_channels * 4, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm1d(out_channels * 4)
        self.dropout = nn.Dropout(dropout_rate)

        self.downsample = None
        if stride != 1 or in_channels != out_channels * 4:
            self.downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels * 4),
            )

    def forward(self, x):
        identity = x
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = F.leaky_relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.dropout(F.leaky_relu(out))

        return out


class ResNet1D(nn.Module):
    def __init__(self, block, layers, num_classes=10, in_channels=1, dropout_rate=0.5):
        super(ResNet1D, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dropout_rate=dropout_rate
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dropout_rate=dropout_rate
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dropout_rate=dropout_rate
        )

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * 4 if block == Bottleneck else 512, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def _make_layer(self, block, out_channels, blocks, stride=1, dropout_rate=0.5):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, dropout_rate))
        self.in_channels = out_channels * 4 if block == Bottleneck else out_channels
        for _ in range(1, blocks):
            layers.append(
                block(self.in_channels, out_channels, dropout_rate=dropout_rate)
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


def resnet50_1d(num_classes=10, in_channels=1, dropout_rate=0.5):
    return ResNet1D(
        Bottleneck,
        [3, 4, 6, 3],
        num_classes=num_classes,
        in_channels=in_channels,
        dropout_rate=dropout_rate,
    )
