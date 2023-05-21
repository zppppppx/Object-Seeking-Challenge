# import sys
# sys.path.append('/path/to/transformer/module')

from ultralytics import YOLO

from models import *
# import models.transformer as tr
# from torchvision.models import ResNet50_Weights

# import gensim as gs

## Test on transformer
# ts = transformer.Transformer(1, 1, 128, 2)
# src = torch.ones(1, 64, 128)
# tgt = torch.ones(1, 10, 128)

# out, memory = ts(src, tgt)
# print(out.shape, memory.shape)

# # Test on resnet
# res = resnet50(ResNet50_Weights.IMAGENET1K_V2)
# x = torch.ones((1,3, 640, 480))
# y = res(x)
# print(y.shape)

## Test on yolo
# yolo = YOLO('./yolov8m.pt')
# x = torch.ones((1,3, 640, 480))
# y = yolo(x)
# print(y)




## other
import os
for dir, dirname, filenames in os.walk('./'):
    print(dir, filenames)
