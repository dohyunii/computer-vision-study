import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torchvision.transforms import v2
from torchvision.io import decode_image

import matplotlib.pyplot as plt

class PreActResidualUnit(nn.Module):
  def __init__(self, in_channels: int, planes: int, stride: int = 1, bottleneck=True):
    super().__init__()
    expansion = 4 if bottleneck else 1
    out_channels = planes * expansion
    self.bn1 = nn.BatchNorm2d(in_channels)
    self.relu = nn.ReLU(inplace=True)

    self.shortcut = (
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        if stride != 1 or in_channels != out_channels
        else None
    )

    if bottleneck:
      self.conv1 = nn.Conv2d(in_channels, planes, 1, stride=stride, bias=False)
      self.bn2 = nn.BatchNorm2d(planes)
      self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
      self.bn3 = nn.BatchNorm2d(planes)
      self.conv3 = nn.Conv2d(planes, out_channels, 1, bias=False)
    else:
      self.conv1 = nn.Conv2d(in_channels, planes, 3, stride=stride, padding=1, bias=False)
      self.bn2 = nn.BatchNorm2d(planes)
      self.conv2 = nn.Conv2d(planes, out_channels, 3, padding=1, bias=False)
      self.conv3 = None

  def forward(self, x):
    preact = self.relu(self.bn1(x))

    identity = (x if self.shortcut is None else self.shortcut(preact))

    out = self.conv1(preact)
    out = self.conv2(self.relu(self.bn2(out)))

    if self.conv3 is not None:
      out = self.conv3(self.relu(self.bn3(out)))

    return identity + out

class PreActResNet(nn.Sequential):
  def __init__(self, layers: list, stage_planes=(64, 128, 256, 512), stem_channels=64, stem_kernel=7, stem_stride=2, max_pool=True, bottleneck=True, num_classes=1000):
    super().__init__()

    self.bottleneck = bottleneck
    self.in_channels = stem_channels

    self.stem = nn.Sequential(
        nn.Conv2d(3, stem_channels, kernel_size=stem_kernel, stride=stem_stride, padding=(stem_kernel - 1) // 2, bias=False),
        nn.MaxPool2d(3, stride=2, padding=1) if max_pool else nn.Identity()
    )
    self.stages = nn.Sequential(*[
        self.make_stage(planes, n_blocks, stride=1 if idx == 0 else 2)
        for idx, (planes, n_blocks)
        in enumerate(zip(stage_planes, layers))
    ])

    self.bn = nn.BatchNorm2d(self.in_channels)
    self.relu = nn.ReLU(inplace=True)

    self.avgpool = nn.AdaptiveAvgPool2d(1)
    self.flatten = nn.Flatten(1)
    self.fc = nn.Linear(self.in_channels, num_classes)

  def make_stage(self, planes, n_blocks, stride):
    expansion = 4 if self.bottleneck else 1

    blocks = [PreActResidualUnit(self.in_channels, planes, stride=stride, bottleneck=self.bottleneck)]
    out_channels = planes * expansion
    self.in_channels = out_channels

    for _ in range(n_blocks - 1):
      blocks.append(PreActResidualUnit(out_channels, planes, stride=1, bottleneck=self.bottleneck))

    return nn.Sequential(*blocks)
