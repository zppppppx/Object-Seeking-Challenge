import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO

class SeekingController(nn.Module):
    def __init__(self,
                imageNet,                       # the class module for image feature extraction
                imageNetPath,                   # the net path for the image feature extraction
                ODNet,                          # the module for the object detector
                ODNetPath,                      # the net path for the object detector
                enc_dec,                        # the module class for encoder and decoder (here we use transformer)
                ) -> None:
        super(SeekingController, self).__init__()

        self.feature_extractor = imageNet(imageNetPath)
        self.object_detector = ODNet(ODNetPath)
        
        self.enc_dec = enc_dec()

