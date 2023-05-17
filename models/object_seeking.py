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
                decision_space=6,               # how many decisions we will make in the end
                n_simulation=1,                 # how many simulations run simultaneously
                memory_length=64,               # how many frames we want to memorize
                feature_length=1000,            # the output of imageNet length
                **enc_dec_params,               # the params needed by encoder and decoder module 
                ) -> None:
        super(SeekingController, self).__init__()

        self.dmodel = enc_dec_params['dmodel']
        self.memory_length = memory_length
        self.decision_space = decision_space

        self.feature_extractor = imageNet(imageNetPath) # let the dimension match with 
        self.feature_matcher = nn.Linear(feature_length, self.dmodel)

        self.object_detector = ODNet(ODNetPath)
        
        # print(enc_dec_params)

        self.enc_dec = enc_dec(**enc_dec_params)

        self.memory = None
        self.pointer = 0


    def forward(self,
                rgb: torch.Tensor,              # input rgb image
                tgt: str,                       # input target (which object to find)
                ):
        bs = rgb.shape[0]                       # batch size
        image_feature = self.feature_matcher(self.feature_extractor(rgb))
        memory = self._padding_with_input(image_feature)

        decision_query = torch.zeros((bs, self.decision_space, self.dmodel))
        output, enc_memory = self.enc_dec(decision_query, memory)


        print(output.shape, enc_memory.shape)
        

    def _padding_with_input(self, memoree: torch.Tensor):
        """
        padding the memory to expected length (memory_length) with input memoree (that needs to be memorized)
        
        Args:
            memoree: the input memoree that will be concanated to the self.memory, should have the shape [b, dmodel]
        """
        memoree = memoree.unsqueeze(1)
        if self.memory is None:
            self.memory = memoree

        # print(memoree.shape, self.memory.shape)

        self.memory = torch.cat((self.memory, memoree), 1)
        if(self.memory.shape[1] > self.memory_length):
            self.memory = self.memory[:, 1:, ...]
            return self.memory
        elif self.memory.shape[1] < self.memory_length:
            patch_size = self.memory_length - self.memory.shape[1]
            patch = torch.zeros_like(memoree).repeat(1, patch_size, 1)

            memory = torch.cat((self.memory, patch), 1)
            return memory

        return self.memory  
            

