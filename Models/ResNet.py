import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torchvision.transforms import v2
from torchvision.io import decode_image

import matplotlib.pyplot as plt

class ResidualUnit(nn.Module):
  def __init__(self, in_channels: int, planes: int, stride: int = 1, bottleneck=True):
    super().__init__()

    expansion = 4 if bottleneck else 1
    out_channels = planes * expansion

    self.relu = nn.ReLU(inplace=True)

    if bottleneck:
        self.conv1 = nn.Conv2d(in_channels, planes, 1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
    else:
        self.conv1 = nn.Conv2d(in_channels, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = None

    self.shortcut = (
        nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1,stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        if stride != 1 or in_channels != out_channels
        else nn.Identity()
    )

  def forward(self, x):
    identity = self.shortcut(x)

    out = self.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))

    if self.conv3 is not None:
      out = self.relu(out)
      out = self.bn3(self.conv3(out))

    return self.relu(identity + out)

class ResNet(nn.Sequential):
  def __init__(self, layers: list, stage_planes=(64, 128, 256, 512), stem_channels=64, stem_kernel=7, stem_stride=2, max_pool=True, bottleneck=True, num_classes=1000):
    super().__init__()

    self.bottleneck = bottleneck
    self.in_channels = stem_channels

    self.stem = nn.Sequential(
        nn.Conv2d(3, stem_channels, kernel_size=stem_kernel, stride=stem_stride, padding=(stem_kernel - 1) // 2, bias=False),
        nn.BatchNorm2d(stem_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(3, stride=2, padding=1)
        if max_pool else nn.Identity()
    )

    self.stages = nn.Sequential(*[
        self.make_stage(planes, n_blocks, stride=1 if idx == 0 else 2) for idx, (planes, n_blocks) in enumerate(zip(stage_planes, layers))
    ])

    self.avgpool = nn.AdaptiveAvgPool2d(1)
    self.flatten = nn.Flatten(1)
    self.fc = nn.Linear(self.in_channels, num_classes)

  def make_stage(self, planes, n_blocks, stride):
    expansion = 4 if self.bottleneck else 1
    out_channels = planes * expansion

    blocks = [ResidualUnit(self.in_channels, planes, stride=stride, bottleneck=self.bottleneck)]

    self.in_channels = out_channels

    for _ in range(n_blocks - 1):
        blocks.append(ResidualUnit(out_channels, planes, stride=1, bottleneck=self.bottleneck))
    return nn.Sequential(*blocks)
