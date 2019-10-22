import torch
from torch import nn
import torch.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

from torch.utils.data import Dataset, DataLoader
from fra_eng_dataset import FraEngDataset, fra_eng_dataset_collate
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

class SelfAttentionHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.d_model = d_model
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
        self.Q = nn.Linear(d_model, d_model)

    def forward(self, x, padding_mask = None, subsq_mask = None):
        # src shape: [N, SEQ, D_MODEL]

        #SelfAttention:
        keys = self.K.forward(x)
        values = self.V.forward(x)
        queries = self.Q.forward(x)

        sqrt_d = self.d_model ** 0.5

        att = torch.matmul(queries, keys.transpose(1,2)) / sqrt_d
        # att shape: [N, SEQ, SEQ]
        # Broadcast padding mask to word attentions so that word attention does not attend to positions outside the sentence
        if padding_mask is not None:
            att = att + padding_mask.transpose(1,2)
        # Add subsequent mask so that each position can attend only itself and the previous elements
        if subsq_mask is not None:
            att = att + subsq_mask.unsqueeze(0)

        att_softmax = torch.softmax(att, dim=2)
        # shape: [N, SEQ, SEQ]
        att_out = torch.matmul(att_softmax, values)
        # shape: [N, SEQ, D_MODEL]

        return att_out, keys, values

class MemAttentionHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.d_model = d_model
        self.Q = nn.Linear(d_model, d_model)

    def forward(self, x, mem_padding_mask, subsq_mask, keys = None, values = None):

        # X shape: [N, SEQ, D_MODEL]

        queries = self.Q.forward(x)

        sqrt_d = self.d_model ** 0.5

        att = torch.matmul(queries, keys.transpose(1,2)) / sqrt_d
        # att shape: [N, SEQ_TGT, SEQ_SRC]

        # Broadcast padding mask to word attentions so that word attention does not attend to positions outside the source sentence
        if mem_padding_mask is not None:
            att = att + mem_padding_mask.transpose(1,2)

        att_softmax = torch.softmax(att, dim=2)
        # shape: [N, SEQ_TGT, SEQ_SRC]
        att_out = torch.matmul(att_softmax, values)
        # shape: [N, SEQ_TGT, D_MODEL]

        return att_out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.num_heads = num_heads
        self.heads = nn.ModuleList([SelfAttentionHead(d_model) for i in range(num_heads)])
        self.linear = nn.Linear(num_heads * d_model, d_model)

    def forward(self, src, src_padding_mask, src_subsq_mask):

        out_cat = None
        keys = None
        values = None

        for i in range(self.num_heads):
            head_outp, keys, values = self.heads[i].forward(src, src_padding_mask, src_subsq_mask)

            if i == 0:
                out_cat = head_outp
            else:
                out_cat = torch.cat([out_cat, head_outp], dim=2)

        ret = self.linear.forward(out_cat)

        return ret, keys, values


class MultiHeadMemAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.num_heads = num_heads
        self.heads = nn.ModuleList([MemAttentionHead(d_model) for i in range(num_heads)])
        self.linear = nn.Linear(num_heads * d_model, d_model)

    def forward(self, src, src_padding_mask, src_subsq_mask, keys, values):

        out_cat = None
        for i in range(self.num_heads):
            head_outp = self.heads[i].forward(src, src_padding_mask, src_subsq_mask, keys = keys, values = values)

            if i == 0:
                out_cat = head_outp
            else:
                out_cat = torch.cat([out_cat, head_outp], dim=2)

        ret = self.linear.forward(out_cat)

        return ret

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_att_heads, ff_dim = 2048, dropout = 0.1):
        super().__init__()

        self.multihead_attention = MultiHeadSelfAttention(d_model, num_att_heads)
        self.att_sublayer_norm = torch.nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, ff_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(ff_dim, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.lin_sublayer_norm = torch.nn.LayerNorm(d_model)

    def forward(self, src, src_padding_mask, src_subsq_mask):

        residual_1 = src
        x, keys, values = self.multihead_attention.forward(src, src_padding_mask, src_subsq_mask)
        x = self.att_sublayer_norm.forward(x + self.dropout1(residual_1))

        residual_2 = x
        x = self.linear2(self.relu(self.linear1.forward(x)))
        x = self.lin_sublayer_norm(x + self.dropout2(residual_2))

        return x, keys, values


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_att_heads, ff_dim = 2048, dropout = 0.1):
        super().__init__()

        self.multihead_self_attention = MultiHeadSelfAttention(d_model, num_att_heads)
        self.self_att_sublayer_norm = torch.nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.multihead_mem_attention = MultiHeadMemAttention(d_model, num_att_heads)
        self.mem_att_sublayer_norm = torch.nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, ff_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(ff_dim, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.lin_sublayer_norm = torch.nn.LayerNorm(d_model)

    def forward(self, x, src_padding_mask, src_subsq_mask, tgt_padding_mask, tgt_subsq_mask, mem_keys, mem_values):

        residual_1 = x
        x, keys, values = self.multihead_self_attention.forward(x, tgt_padding_mask, tgt_subsq_mask)
        x = self.self_att_sublayer_norm.forward(x + self.dropout1(residual_1))

        residual_2 = x
        x = self.multihead_mem_attention.forward(x, src_padding_mask, tgt_subsq_mask, keys = mem_keys, values = mem_values)
        x = self.mem_att_sublayer_norm.forward(x + self.dropout2(residual_2))

        residual_3 = x
        x = self.linear2(self.relu(self.linear1.forward(x)))
        x = self.lin_sublayer_norm(x + self.dropout3(residual_3))

        return x


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_att_heads):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_att_heads) for i in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self,src, src_padding_mask, src_subsq_mask):
        x = src

        keys = None
        values = None
        for layer in self.layers:
            x, keys, values = layer.forward(x, src_padding_mask, src_subsq_mask)

        x = self.norm.forward(x)

        return keys, values


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_att_heads):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_att_heads) for i in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, src_padding_mask, src_subsq_mask, tgt_padding_mask, tgt_subsq_mask, mem_keys, mem_values):
        x = tgt
        for layer in self.layers:
            x = layer.forward(x, src_padding_mask, src_subsq_mask, tgt_padding_mask, tgt_subsq_mask, mem_keys, mem_values)

        x = self.norm.forward(x)

        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        self.sin_args = torch.zeros(1, self.d_model).to(device)
        self.cos_args = torch.zeros(1, self.d_model).to(device)
        for i in range(self.d_model//2):
            self.sin_args[0,i * 2] = 10000**(2.*i/self.d_model)
            self.cos_args[0,i * 2 + 1] = 10000**(2.*i/self.d_model)

        self.sin_args_filled = (self.sin_args > 1e-10).float()
        self.sin_args = self.sin_args + (self.sin_args < 1e-10).float()

        self.cos_args_filled = (self.cos_args > 1e-10).float()
        self.cos_args = self.cos_args + (self.cos_args < 1e-10).float()

    def forward(self, x):
        for pos in range(x.size()[-2]):
            x[:,pos,:] = x[:,pos,:] + \
                         torch.sin(pos / self.sin_args) * self.sin_args_filled + \
                         torch.cos(pos / self.cos_args) * self.cos_args_filled

        return x

# positional_enc = PositionalEncoding(256)
# data = torch.zeros(1, 50, 256)
# data_pos_enc = positional_enc.forward(data)
#
# enc_np = data_pos_enc.squeeze(dim=0).numpy()
# plt.imshow(enc_np)
# plt.show()

class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_att_heads, input_dict_size, output_dict_size):
        super().__init__()

        self.input_emb = nn.Embedding(input_dict_size, d_model)
        self.outp_emb = nn.Embedding(output_dict_size, d_model)

        self.positional_encoder = PositionalEncoding(d_model)
        self.encoder = Encoder(num_layers, d_model, num_att_heads)
        self.decoder = Decoder(num_layers, d_model, num_att_heads)

        self.outp_logits = nn.Linear(d_model, output_dict_size)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, src, tgt, src_padding_mask, src_subsq_mask, tgt_padding_mask, tgt_subsq_mask):

        enc_x = self.input_emb.forward(src.squeeze(dim=2))
        enc_x = self.positional_encoder.forward(enc_x)
        enc_keys, enc_values = self.encoder.forward(enc_x, src_padding_mask, src_subsq_mask)

        dec_x = self.outp_emb.forward(tgt.squeeze(dim=2))
        dec_x = self.decoder.forward(dec_x, src_padding_mask, src_subsq_mask, tgt_padding_mask, tgt_subsq_mask, enc_keys, enc_values)
        dec_x = self.outp_logits.forward(dec_x)
        dec_x = self.softmax(dec_x)

        return dec_x


BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 100
STORE_MODELS = True
models_path = 'models'

if not os.path.exists(models_path):
    os.mkdir(models_path)


dataset = FraEngDataset()
sentences_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=fra_eng_dataset_collate)

in_dict_size = dataset.get_fra_dict_size()
out_dict_size = dataset.get_eng_dict_size()


transformer_model = Transformer(
    num_layers=6,
    d_model=256,
    num_att_heads=8,
    input_dict_size=in_dict_size,
    output_dict_size=out_dict_size
).to(device)


def get_square_subsequent_mask(seq_len):
    mask = (torch.triu(torch.ones(seq_len, seq_len).to(device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def get_padding_mask(input, val1 = float('-inf'), val2 = float(0.0)):
    mask = torch.ones(input.size()).to(device)
    mask = mask.float().masked_fill(input == 0, val1).masked_fill(input > 0, val2)
    return mask


def get_one_hot(x, out_dim, mask):

    tens = x.view(-1)
    tens_one_hot = torch.zeros(list(tens.size()) + [out_dim]).to(device)
    for i in range(len(tens)):
        tens_one_hot[i,tens[i]] = 1

    tens_one_hot = tens_one_hot.view(list(x.size()) + [out_dim])
    tens_one_hot = tens_one_hot * mask
    return tens_one_hot.to(device)


optimizer = torch.optim.Adam(transformer_model.parameters(), lr = 1e-4)

iterations = 0

for epoch in range(EPOCHS):

    train_loss_sum = 0.0
    total_word_count = 0.0

    for sentences in sentences_loader:

        src_sentences = sentences['fra_sentences']
        tgt_sentences = sentences['eng_sentences']

        # Create tensors from token lists
        padded_src = pad_sequence(src_sentences, padding_value=0, batch_first=True).to(device)
        padded_tgt = pad_sequence(tgt_sentences, padding_value=0, batch_first=True).to(device)

        src_padding_mask = get_padding_mask(padded_src)
        src_subsq_mask = get_square_subsequent_mask(padded_src.size()[1])

        tgt_padding_mask = get_padding_mask(padded_tgt)
        tgt_subsq_mask = get_square_subsequent_mask(padded_tgt.size()[1])

        pred = transformer_model.forward(
            src=padded_src,
            tgt=padded_tgt,
            src_padding_mask=src_padding_mask,
            src_subsq_mask=src_subsq_mask,
            tgt_padding_mask=tgt_padding_mask,
            tgt_subsq_mask=tgt_subsq_mask
        )

        # Mask to zero one hot vectors corresponding to padded elements
        one_hot_mask = get_padding_mask(padded_tgt, val1=float(0.0), val2=float(1.0))
        y_one_hot = get_one_hot(padded_tgt.squeeze(dim=2), out_dict_size, mask=one_hot_mask)

        loss = - torch.sum(torch.log(pred) * y_one_hot)
        print(loss)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss_sum += loss.detach().to('cpu').data
        total_word_count += torch.sum(y_one_hot).to('cpu').data

    print(f"Epoch {epoch} " + '=' * 60)
    print(f"Total loss per word: {train_loss_sum / total_word_count}")
    print(f"Some generated sentences:")

    if STORE_MODELS == True:
        model_path = os.path.join(models_path, f'Epoch_{epoch}_model.pt')
        torch.save(transformer_model, model_path)

print("Traing done! Generating some more sentences.")
