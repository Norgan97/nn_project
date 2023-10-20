import torchutils as tu
import torch
import torchvision
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T
from torchvision import io
from torchvision.models import resnet18, ResNet18_Weights
model_two = resnet18(weights = ResNet18_Weights.DEFAULT)
for param in model_two.parameters():
    param.requires_grad = False

model_two.fc = nn.Linear(512,1)