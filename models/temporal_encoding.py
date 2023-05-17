import torch
import torch.nn as nn
import math

class TemporalEncoding(nn.Module):
    def __init__(self, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x: torch.Tensor):
        shape = x.shape
        frame_num = shape[1]                                # the number of the memorized frames
        dmodel = shape[2]                                   # the dimension of the features
        embedding = torch.ones(shape, device=x.device)
        embedding_t = embedding.cumsum(1) - 1
        embedding_feat = embedding.cumsum(2)
        
        if self.normalize:
            embedding_t = embedding_t / frame_num * self.scale

            
        dim_t = self.temperature**(embedding_feat//2 * 2 / dmodel)
        temp_enc = embedding_t / dim_t
        temp_enc[..., 0::2] = temp_enc[..., 0::2].sin()
        temp_enc[..., 1::2] = temp_enc[..., 1::2].cos()


        return temp_enc
        
