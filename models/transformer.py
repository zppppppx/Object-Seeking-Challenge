import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class Transformer(nn.Module):
    def __init__(self,
                encoder_num,
                decoder_num,
                dmodel,                     # the input dimensions of the features, also used in positional encoding
                nhead=8,                        # the number of multi-heads in the multi-head attention
                dim_ffn=512,                    # the hidden dimensions of ffn
                dropout=0.1,                    # dropout prob
                activation=F.relu,              # activation function
                return_intermediate=False       # whether return intermediate results (the results from multiple decoders)
                ) -> None:
        super(Transformer, self).__init__()

        encoder_layer = EncoderLayer(dmodel, nhead, dim_ffn, dropout, activation)
        self.encoder = Encoder(encoder_layer, encoder_num)

        decoder_layer = DecoderLayer(dmodel, nhead, dim_ffn, dropout, activation)
        self.decoder = Decoder(decoder_layer, decoder_num)

        self.return_intermediate = return_intermediate

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, 
                tgt,                            # the target (decoder's input) should have the shape [batch x output_length x feature_length]
                src,                            # the source (encoder's input) should have the shape [batch x memory_length x feature_length]
                src_mask=None,                  # used to indicate which parts are padding blocks (in this project, it's not likely to be used)
                src_temp_encoding=None,         # the temporal encoding for src (encoder's input)
                query_temp_encoding=None        # the temporal encoding for query (decoder's input)
                ):
        
        memory = self.encoder(src, src_mask, src_temp_encoding)
        out = self.decoder(tgt, memory, src_mask, src_temp_encoding, query_temp_encoding)

        return out.transpose(1, 0), memory

        


class EncoderLayer(nn.Module):
    def __init__(self, dmodel, nhead, dim_ffn=512, dropout=0.1, activation=F.relu) -> None:
        super(EncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(dmodel, nhead, dropout=dropout, batch_first=True)

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

    def with_temp_embed(self, tensor, temp=None):
        return tensor if temp is None else tensor + temp

    def forward(self, 
                src,
                src_mask = None,
                src_key_padding_mask = None,
                temp = None
                ):
        q = k = self.with_temp_embed(src, temp)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers=3) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for i in range(num_layers)])
        
    def forward(self, src,
                mask = None,
                src_key_padding_mask = None,
                temp = None):
        output = src

        for layer in self.layers:
            output = layer(output, mask, src_key_padding_mask, temp)

        return output


class DecoderLayer(nn.Module):
    def __init__(self, dmodel, nhead, dim_ffn=512, dropout=0.1, activation=F.relu) -> None:
        super(DecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(dmodel, nhead, dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dmodel, nhead, dropout, batch_first=True)

        # FFN
        self.linear1 = nn.Linear(dmodel, dim_ffn)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_ffn, dmodel)

        self.norm1 = nn.LayerNorm(dmodel)
        self.norm2 = nn.LayerNorm(dmodel)
        self.norm3 = nn.LayerNorm(dmodel)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = activation

    def with_temp_embed(self, tensor, temp=None):
        return tensor if temp is None else tensor + temp
    
    def forward(self, 
                tgt,                            # the input of the decoder
                memory,                         # the output of the encoder (also the input of the cross attention block)
                tgt_mask = None,
                memory_mask = None,
                tgt_key_padding_mask = None,
                memory_key_padding_mask = None,
                temp = None,                    # temporal encoding of decoder's input
                temp_memory = None,             # temporal encoding of encoder's output
                ):
        q = k = self.with_temp_embed(tgt, temp)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.cross_attn(query=self.with_temp_embed(tgt, temp),
                               key=self.with_temp_embed(memory, temp_memory),
                               value = memory, attn_mask=memory_mask,
                               key_padding_mask = memory_key_padding_mask)[0]
        
        # Second add-norm layer
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Third add-norm layer
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt
    

class Decoder(nn.Module):
    def __init__(self, decoder_layer, num_layers=3, return_intermediate=False) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for i in range(num_layers)])
        self.return_intermediate = return_intermediate
        
    def forward(self,
                tgt,                            # the input of the decoder
                memory,                         # the output of the encoder (also the input of the cross attention block)
                tgt_mask = None,
                memory_mask = None,
                tgt_key_padding_mask = None,
                memory_key_padding_mask = None,
                temp = None,                    # temporal encoding of decoder's input
                temp_memory = None,             # temporal encoding of encoder's output
                ):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask, memory_mask, tgt_key_padding_mask, 
                           memory_key_padding_mask, temp, temp_memory)
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)
        

        return output.unsqueeze(0)
