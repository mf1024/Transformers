import torch
from torch import nn

class SelfAttentionHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.d_model = d_model

        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
        self.Q = nn.Linear(d_model, d_model)

        self.FF = nn.Linear(d_model, d_model)

    def forward(self, x, mask):
        # X shape: [N, SEQ, D_MODEL]

        #SelfAttention:
        keys = self.K.forward(x)
        values = self.V.forward(x)
        queries = self.Q.forward(x)

        sqrt_d = self.d_model ** 0.5

        att = torch.matmul(queries, keys.transpose(1,2)) / sqrt_d
        # shape: [N, SEQ, SEQ]
        att_softmax = torch.softmax(att, dim=1)
        # shape: [N, SEQ, SEQ]
        att_out = torch.matmul(att_softmax, values)
        # shape: [N, SEQ, D_MODEL]

        return att_out


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.num_heads = num_heads
        self.heads = nn.ModuleList([SelfAttentionHead(d_model) for i in range(num_heads)])
        self.linear = nn.Linear(num_heads * d_model, d_model)

    def forward(self, x):

        out_cat = None
        for i in range(self.num_heads):
            if i == 0:
                out_cat = self.heads[i].forward(x)
            else:
                out_cat = torch.cat(out_cat, self.heads[i], dim=2)

        ret = self.linear.forward(out_cat)

        return ret

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_att_heads):
        super().__init__()

        self.multihead_attention = MultiHeadAttention(d_model, num_att_heads)
        self.att_layer_norm = torch.nn.LayerNorm(d_model)

        self.linear = nn.Linear(d_model, d_model)
        self.lin_layer_norm = torch.nn.LayerNorm(d_model)

    def forward(self, x):

        res1 = x
        x = self.multihead_attention.forward(x)
        x = self.att_layer_norm.forward(x + res1)

        res2 = x
        x = self.linear.forward(x)
        x = self.lin_layer_norm(x + res2)

        return x


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_att_heads):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_att_heads) for i in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)

        return x

class PositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pos):



class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_att_heads):
        super().__init__()


        #TODO: masking, positional encoding

        self.positional_encoder = None
        self.encoder = Encoder(num_layers, d_model, num_att_heads)
        self.decoder = None

    def forward(self, x):
        x = self.encoder(x)








