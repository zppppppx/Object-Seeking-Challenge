import torch
import torch.nn as nn
import torch.nn.functional as F
from temporal_encoding import TemporalEncoding

class Transformer(nn.Module):
    def __init__(self,
                encoder_num,
                decoder_num,
                memory_length=64,               # how many previous frames should be memorized
                dmodel=256,                     # the input dimensions of the features, also used in positional encoding
                temperature=10000,              # the constant used in the positional encoding
                ) -> None:
        super(Transformer, self).__init__()
        
        self.dmodel = dmodel
        self.temperature = temperature
        self.memory_length = memory_length

class EncoderLayer(nn.Module):
    def __init__(self, dmodel, nhead, dim_ffn=512, dropout=0.1, activation=F.relu, temporal_encoding=True) -> None:
        super(EncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(dmodel, nhead, dropout=dropout)

        # FFN
        self.linear1 = nn.Linear(dmodel, dim_ffn)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_ffn, dmodel)

        # Add and norm
        self.norm1 = nn.LayerNorm(dmodel)
        self.norm2 = nn.LayerNorm(dmodel)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = activation

        # Temporal encoding
        self.temporal_encoder = TemporalEncoding()

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward(self, 
                src,
                src_mask = None,
                src_key_padding_mask = None,
                pos = None
                ):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src



class DecoderLayer(nn.Module):
    def __init__(self, dmodel) -> None:
        super(DecoderLayer, self).__init__()
        self.dmodel = dmodel