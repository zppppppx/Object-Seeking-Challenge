from ultralytics import YOLO
from models import *
from torchvision.models import ResNet50_Weights

# Test on object seeking
os = object_seeking.SeekingController(resnet50.ResNet50, ResNet50_Weights.IMAGENET1K_V2,
                                      YOLO, './yolov8m.pt',
                                      transformer.Transformer,
                                      6, 1, 64, 1000, encoder_num=3, decoder_num=3, dmodel=512)

x = torch.ones((3, 3, 640, 480))
y = os(x, "hah")

