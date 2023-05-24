import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from utils.word2vec.w2c import *

from .temporal_encoding import TemporalEncoding


class Reshape(nn.Module):
    def __init__(self, *shape) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view((x.size(0),)+self.shape)

class SeekingController(nn.Module):
    def __init__(self,
                imageNet,                       # the class module for image feature extraction
                imageNetPath,                   # the net path for the image feature extraction
                ODNet,                          # the module for the object detector
                ODNetPath,                      # the net path for the object detector
                enc_dec,                        # the module class for encoder and decoder (here we use transformer)
                wv_path,                        # the word2vec file path
                decision_space=128,             # the dimensions of the decision space
                n_simulation=1,                 # how many simulations run simultaneously
                memory_length=64,               # how many frames we want to memorize
                feature_length=1000,            # the output of imageNet length
                context_grid_shape=(32, 32),    # the shape of the context grid map
                fo_conv_layers=3,               # the number of conv layers to extract the feature of object grid map
                **enc_dec_params,               # the params needed by encoder and decoder module 
                ) -> None:
        super(SeekingController, self).__init__()

        self.dmodel = enc_dec_params['dmodel']
        self.memory_length = memory_length
        self.decision_space = decision_space

        self.feature_extractor = imageNet(imageNetPath) # let the dimension match with 
        self.fv = nn.Linear(feature_length, self.dmodel//2) # the vision learner

        self.object_detector = ODNet(ODNetPath)
        self.fo = self._build_fo(fo_conv_layers, context_grid_shape, self.dmodel//2)

        # the feature fusion layer
        self.fm = nn.Sequential(
            nn.Linear(self.dmodel, self.dmodel),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True)
        )
        
        # print(enc_dec_params)

        self.enc_dec = enc_dec(**enc_dec_params)

        self.memory = None
        self.pointer = 0

        self.wv = load_vector(wv_path) # the word vector
        self.contex_grid_shape = context_grid_shape

        self.tmpEnc = TemporalEncoding() # the temporal encoder

        self.final_decision_layer = nn.Sequential(
            nn.Linear(self.dmodel, decision_space*2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(decision_space*2, decision_space)
        )



    def forward(self,
                rgb: torch.Tensor,              # input rgb image
                tgt: str,                       # input target (which object to find)
                ):
        bs = rgb.shape[0]                       # batch size
        shape = rgb.shape[2:]                   # picture size (h, w)

        image_feature = self.fv(self.feature_extractor(rgb))

        detections = self.object_detector(rgb)
        masks = self._detection_mask(detections, bs, self.contex_grid_shape, tgt)

        object_grid_feature = self.fo(masks)
        features = torch.cat((image_feature, object_grid_feature), dim=1)

        features = self.fm(features)

        memory = self._padding_with_input(features)
        memory_tmpEnc = self.tmpEnc(memory)

        # decision_query = torch.zeros((bs, self.decision_space, self.dmodel))
        decision_query = self._get_word_vector(bs, tgt)
        
        state_vector, enc_memory = self.enc_dec(decision_query, memory, src_temp_encoding=memory_tmpEnc)
        
        state_vector = state_vector[:, 0, 0, ...] # left bs x dimension
        print(state_vector.shape)

        state_vector = self.final_decision_layer(state_vector)
        print(state_vector.shape)

        return state_vector
        

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
            
    def _detection_mask(self, detections, bs, shape, tgt):
        """
        Using the detections and word vector to build a similarity masks

        Args:   
            detections: output from the object detector
            bs: batch_size
            shape: the shape of the context grid
            tgt: the target we want to find

        Returns:
            masks: a similarity mask with shape: [bs, 1, shape[0], shape[1]]
        """
        masks = torch.zeros((bs, 1, shape[0], shape[1]))
        # print(masks.shape)
        for i in range(bs):
            for box in detections[i].boxes:
                # print(box)
                cls = detections[i].names[int(box.cls)]
                similarity_score = similarity(self.wv, cls, tgt)
                box_loc = box.xywh.int()
                # print(box_loc)
                masks[i, 0, box_loc[0, 1]//shape[0], box_loc[0, 0]//shape[1]] = \
                    torch.max(torch.tensor(similarity_score), masks[i, 0, box_loc[0, 1]//shape[0], box_loc[0, 0]//shape[1]])

        return masks


    def _build_fo(self, num_layer, context_grid_shape, feature_length):
        """
        Build a learnable layer fo according to the shape and expected number of layers

        Args:
            num_layer: how many conv2d layers will be used
            context_grid_shape: the shape of the context grid map, note that we assume only square maps are allowed here
            feature_length: the expected feature length of the output, the final output shape should be [bs x feature_length]

        Returns:
            fo: the constructed feature extractor of context grid map.
        """
        layers = []
        in_channel = 1
        out_channel = 64
        final_size = int(context_grid_shape[0] / pow(2, num_layer)) ** 2

        assert final_size > 0, "The shape is incorrect, try to reduce number of layers or increase the context grid map size"

        for i in range(num_layer):
            layers.extend([
                nn.Conv2d(in_channel, out_channel, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
                ])
            in_channel = out_channel
            out_channel *= 2

        layers.extend([
            Reshape(final_size * in_channel),
            nn.Linear(final_size * in_channel, feature_length),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True)
            ])
        
        return nn.Sequential(*layers)
    
    def _get_word_vector(self, bs, tgt):
        wv = torch.tensor(self.wv[tgt]).view(1, 1, -1)
        wv = wv.repeat(bs, 1, 1)

        # print(wv.shape)

        wv = F.interpolate(wv, size=(self.dmodel), mode='linear')

        return wv
