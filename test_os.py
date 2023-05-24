from ultralytics import YOLO
from models import *
from torchvision.models import ResNet50_Weights
import cv2

# Test on object seeking
os = object_seeking.SeekingController(resnet50.ResNet50, ResNet50_Weights.IMAGENET1K_V2,
                                        YOLO, './yolov8m.pt',
                                        transformer.Transformer, 'F:\grad\quater3\ECE285\Object-Seeking-Challenge\model.txt',
                                        decision_space=128,             # the dimensions of the decision space
                                        n_simulation=1,                 # how many simulations run simultaneously
                                        memory_length=64,               # how many frames we want to memorize
                                        feature_length=1000,            # the output of imageNet length
                                        context_grid_shape=(32, 32),    # the shape of the context grid map
                                        fo_conv_layers=3, 
                                        encoder_num=3, decoder_num=3, dmodel=512)

img = cv2.imread('./test.jpg')
img = cv2.resize(img, (640, 480))
img = torch.tensor(img)
img = img.unsqueeze(0)
img = img.permute(0, 3, 1, 2).float()
# x = torch.ones((3, 3, 480, 640))
y = os(img, "hah")
print(y.shape)

