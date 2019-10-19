import torch
from torch import nn
import torch.functional as F
import numpy as np
import matplotlib.pyplot as plt

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

    def forward(self, src, src_padding_mask, src_subsq_mask):
        # X shape: [N, SEQ, D_MODEL]

        #SelfAttention:
        keys = self.K.forward(src)
        values = self.V.forward(src)
        queries = self.Q.forward(src)

        sqrt_d = self.d_model ** 0.5

        att = torch.matmul(queries, keys.transpose(1,2)) / sqrt_d
        # shape: [N, SEQ, SEQ]
        # Broadcast padding mask to word attentions so that word attention does not attend to positions outside the sentence
        att = att + src_padding_mask.transpose(1,2)
        # Add subsequent mask so that each position can attend only itself and the previous elements
        att = att + src_subsq_mask.unsqueeze(0)
        att_softmax = torch.softmax(att, dim=2)
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

    def forward(self, src, src_padding_mask, src_subsq_mask):

        out_cat = None
        for i in range(self.num_heads):
            if i == 0:
                out_cat = self.heads[i].forward(src, src_padding_mask, src_subsq_mask)
            else:
                out_cat = torch.cat([out_cat, self.heads[i].forward(src, src_padding_mask, src_subsq_mask)], dim=2)

        ret = self.linear.forward(out_cat)

        return ret

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_att_heads, ff_dim = 2048, dropout = 0.1):
        super().__init__()

        self.multihead_attention = MultiHeadAttention(d_model, num_att_heads)
        self.att_sublayer_norm = torch.nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, ff_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(ff_dim, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.lin_sublayer_norm = torch.nn.LayerNorm(d_model)

    def forward(self, src, src_padding_mask, src_subsq_mask):

        res1 = src
        x = self.multihead_attention.forward(src, src_padding_mask, src_subsq_mask)
        x = self.att_sublayer_norm.forward(x + self.dropout1(res1))

        res2 = x
        x = self.linear2(self.relu(self.linear1.forward(x)))
        x = self.lin_sublayer_norm(x + self.dropout2(res2))

        return x


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_att_heads):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_att_heads) for i in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self,src, src_padding_mask, src_subsq_mask):
        x = src
        for layer in self.layers:
            x = layer.forward(x, src_padding_mask, src_subsq_mask)

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

        #TODO: returning memory from encoder and decoder
        #TODO: decoder

        self.input_emb = nn.Embedding(input_dict_size, d_model)

        self.positional_encoder = PositionalEncoding(d_model)
        self.encoder = Encoder(num_layers, d_model, num_att_heads)
        self.decoder = None

        self.outp_logits = nn.Linear(d_model, output_dict_size)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, src, src_padding_mask, src_subsq_mask):

        x = self.input_emb.forward(src.squeeze(dim=2))
        x = self.positional_encoder.forward(x)

        #TODO: for now will use just encoder for language modeling task
        x = self.encoder.forward(x, src_padding_mask, src_subsq_mask)
        x = self.outp_logits.forward(x)
        x = self.softmax(x)

        return x


BATCH_SIZE = 128
LEARNING_RATE = 1e-4
EPOCHS = 100


dataset = FraEngDataset()
sentences_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=fra_eng_dataset_collate)

in_dict_size = dataset.get_eng_dict_size()

transformer_model = Transformer(
    num_layers=6,
    d_model=512,
    num_att_heads=8,
    input_dict_size=in_dict_size,
    output_dict_size=in_dict_size # We do language modeling so we will use in_dict_size for output as well
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

def print_some_outputs(src, pred):

    for i in range(len(src)):
        if i > 10:
            break
        src_seq = torch.squeeze(src[i], dim=1)
        pred_seq = torch.argmax(pred[i], dim=1)

        src_sentence = ''
        for word_idx in src_seq:
            src_sentence += dataset.eng_token_to_text[word_idx] + ' '

        pred_sentence = ''
        for word_idx in pred_seq:
            pred_sentence += dataset.eng_token_to_text[word_idx] + ' '

        print("Source sentence is:")
        print(src_sentence)
        print("Pred sentence is:")
        print(pred_sentence)

def generate_some_sentences(num_sentences = 25):

    transformer_model.eval()

    with torch.no_grad():
        for i in range(num_sentences):
            snt = torch.ones((1,1,1)) * dataset.get_eng_start_code()
            snt = snt.long()
            snt = snt.to(device)

            sent_idxes = []

            for i in range(25):
                pred = transformer_model.forward(
                    src = snt,
                    src_padding_mask = torch.zeros_like(snt).float(),
                    src_subsq_mask = get_square_subsequent_mask(snt.size()[1]),
                )
                next_word_softmax = pred[0,i,:].to('cpu').detach().numpy()
                next_word = np.random.choice(len(next_word_softmax), p=next_word_softmax)
                snt = torch.cat([snt, torch.ones((1,1,1)).long().to(device) * next_word], dim=1)

                sent_idxes.append(next_word)

                if next_word == dataset.get_eng_eos_code():
                    break

            sent = ''
            for word_idx in sent_idxes:
                sent = f"{sent} {dataset.eng_token_to_text[word_idx]}"

            print(sent)

    transformer_model.train()


iterations = 0

for epoch in range(EPOCHS):
    for sentences in sentences_loader:

        # in_sentences = sentences['fra_sentences']
        # in_lens = sentences['fra_lens']
        # out_sentences = sentences['eng_sentences']
        # out_lens = sentences['eng_lens']
        src_sentences = sentences['eng_sentences']
        tgt_sentences = []

        for sentence in src_sentences:
            tgt_sentences.append(sentence[1:])

        for sent_idx in range(len(src_sentences)):
            src_sentences[sent_idx] = src_sentences[sent_idx][:-1]

        padded_src = pad_sequence(src_sentences, padding_value = 0, batch_first=True).to(device)
        padded_tgt = pad_sequence(tgt_sentences, padding_value = 0, batch_first=True).to(device)

        src_padding_mask = get_padding_mask(padded_src)
        src_subsq_mask = get_square_subsequent_mask(padded_src.size()[1])

        pred = transformer_model(
            src = padded_src,
            src_padding_mask = src_padding_mask,
            src_subsq_mask = src_subsq_mask
        )

        iterations = iterations + 1
        if iterations == 100:
            # print_some_outputs(padded_src, pred)
            generate_some_sentences()
            iterations = 0

        #Creating one hot mask to zero one hot vectors corresponding to padded elements
        one_hot_mask = get_padding_mask(padded_tgt, val1 = float(0.0), val2 = float(1.0))

        y_one_hot = get_one_hot(padded_tgt.squeeze(dim=2), in_dict_size, mask = one_hot_mask)

        loss = - torch.sum(torch.log(pred) * y_one_hot)
        print(loss / torch.sum(y_one_hot))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
