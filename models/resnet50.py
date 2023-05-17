import torch.nn as nn
import torch
from torchvision.models import resnet50

class ResNet50(nn.Module):
    """
    Pretrained resnet 50. Input size is b x 3 x H x W, output size is b x 1000  
    """
    def __init__(self, netPath) -> None:
        super(ResNet50, self).__init__()

        self.resnet50 = resnet50(netPath)

    def forward(self, x):
        return self.resnet50(x)

