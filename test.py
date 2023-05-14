# import sys
# sys.path.append('/path/to/transformer/module')

from ultralytics import YOLO

from models import *
import models.transformer as tr

from torchvision.models import ResNet50_Weights

# from modules.transformer import Transformer
osc = object_seeking.SeekingController(resnet50.ResNet50, ResNet50_Weights.IMAGENET1K_V2, YOLO, "./yolov8m.pt", tr.Transformer)

# resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)



# net = resnet50.ResNet50(ResNet50_Weights.IMAGENET1K_V2)
# a = torch.ones((3, 3, 224, 224))
# b = net(a)
# print(b.shape)
