import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torchvision.transforms import v2
from torchvision.io import decode_image

import matplotlib.pyplot as plt

class DenseLayer(nn.Sequential):
  def __init__(self, in_channels, growth_rate, bottleneck):
    if bottleneck:
      expansion = growth_rate * 4
      super().__init__(
          nn.BatchNorm2d(in_channels),
          nn.ReLU(inplace=True),
          nn.Conv2d(in_channels, expansion, kernel_size=1, padding=0, bias=False),

          nn.BatchNorm2d(expansion),
          nn.ReLU(inplace=True),
          nn.Conv2d(expansion, growth_rate, kernel_size=3, padding=1, bias=False)
      )
    else:
      super().__init__(
          nn.BatchNorm2d(in_channels),
          nn.ReLU(inplace=True),
          nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
      )

class DenseBlock(nn.Module):
  def __init__(self, in_channels: int, n_layers: int, growth_rate: int = 16, bottleneck=True):
    super().__init__()

    self.layers = nn.ModuleList([DenseLayer(in_channels + i * growth_rate, growth_rate, bottleneck) for i in range(n_layers)])

  def forward(self, x):
    h = [x]
    for layer in self.layers:
      h.append(layer(torch.cat(h, dim=1)))
    return torch.cat(h, dim=1)

class DenseNet(nn.Sequential):
  def __init__(self, layers: list, growth_rate: int = 16,
               stem_channels=32, stem_kernel=7, stem_stride=2, max_pool=True,
               bottleneck=True, compression_factor: float = 1.0, num_classes=1000):
    super().__init__()

    self.in_channels = stem_channels

    if max_pool:
      self.stem = nn.Sequential(
          nn.Conv2d(3, stem_channels, kernel_size=stem_kernel, stride=stem_stride, padding=(stem_kernel - 1) // 2, bias=False),
          nn.BatchNorm2d(stem_channels),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(3, stride=2, padding=1)
      )
    else:
      self.stem = nn.Conv2d(3, stem_channels, kernel_size=stem_kernel, stride=stem_stride, padding=(stem_kernel - 1) // 2, bias=False)

    self.stages = nn.Sequential()
    for idx, n_blocks in enumerate(layers):
      self.stages.append(DenseBlock(self.in_channels, n_blocks, growth_rate, bottleneck=bottleneck))
      self.in_channels += n_blocks * growth_rate
      if idx != len(layers) - 1: # transition layer
        out_channels = int(compression_factor * self.in_channels)
        self.stages.append(nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True), # ???
            nn.Conv2d(self.in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        ))
        self.in_channels = out_channels

    self.classification = nn.Sequential(
        nn.BatchNorm2d(self.in_channels), # ?
        nn.ReLU(inplace=True), # ?
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(1),
        nn.Linear(self.in_channels, num_classes)
    )
