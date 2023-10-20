import torchutils as tu
import torch
import torchvision
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T
from torchvision import io
from torchvision.models import resnet50, ResNet50_Weights
model_one = resnet50(weights = ResNet50_Weights.DEFAULT)
for param in model_one.parameters():
    param.requires_grad = False

model_one.fc = nn.Linear(2048,6)