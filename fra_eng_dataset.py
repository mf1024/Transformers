from torch.utils.data import Dataset
import pickle
from nltk.tokenize import word_tokenize
import os
import torch
import numpy as np

class FraEngDataset(Dataset):
    def __init__(self, data_source_path = 'fra-eng/fra.txt'):
        super().__init__()
        
        data_file_path = "sentences.pkl"
        self.sentence_list = []
        
        self.eng_token_dict = dict()
        self.eng_token_dict['<PAD>'] = 0
        self.eng_token_dict['<EOS>'] = 1
        self.eng_token_dict['<START>'] = 2
        self.eng_token_count = 2
        self.eng_token_to_text = ['<PAD>', '<EOS>', '<START>']
        
        self.fra_token_dict = dict()
        self.fra_token_dict['<PAD>'] = 0
        self.fra_token_dict['<EOS>'] = 1
        self.fra_token_dict['<START>'] = 2
        self.fra_token_count = 2
        self.fra_token_to_text = ['<PAD>', '<EOS>', '<START>']
        
        
        if os.path.exists(data_file_path):
            with open(data_file_path, 'rb') as f:
                pickle_data = pickle.load(f)
                self.sentence_list = pickle_data['sentence_list']
                self.eng_token_count = pickle_data['eng_token_count']
                self.eng_token_to_text = pickle_data['eng_token_to_text']
                self.fra_token_count = pickle_data['fra_token_count']
                self.fra_token_to_text = pickle_data['fra_token_to_text']
       
        else:
        
            with open(data_source_path, "r", encoding='utf-8') as f:
                for idx, line in enumerate(f.readlines()):

                    eng_token_sentence = []
                    fra_token_sentence = []
                    
                    snt = line.split('\t') 
                    eng_sentence = snt[0]
                    fra_sentence = snt[1]

                    eng_token_list = word_tokenize(eng_sentence)
                    for token in eng_token_list:
                        if token not in self.eng_token_dict:
                            self.eng_token_count += 1
                            self.eng_token_dict[token] = self.eng_token_count
                            self.eng_token_to_text.append(token)
                        
                        token_idx = self.eng_token_dict[token]
                        eng_token_sentence.append(token_idx)

                    eng_token_sentence = [self.eng_token_dict['<START>']] + eng_token_sentence
                    eng_token_sentence.append(self.eng_token_dict['<EOS>'])

                    fra_token_list = word_tokenize(fra_sentence)
                    for token in fra_token_list:
                        if token not in self.fra_token_dict:
                            self.fra_token_count += 1
                            self.fra_token_dict[token] = self.fra_token_count
                            self.fra_token_to_text.append(token)

                        token_idx = self.fra_token_dict[token]
                        fra_token_sentence.append(token_idx)

                    fra_token_sentence = [self.eng_token_dict['<START>']] + eng_token_sentence
                    fra_token_sentence.append(self.fra_token_dict['<EOS>'])
                        
                    self.sentence_list.append(
                        dict(
                            eng = eng_token_sentence,
                            fra = fra_token_sentence
                        ))

            with open(data_file_path, "wb") as f:
                pickle_data = dict(
                    sentence_list = self.sentence_list,
                    eng_token_count = self.eng_token_count,
                    fra_token_count = self.fra_token_count,
                    eng_token_to_text = self.eng_token_to_text,
                    fra_token_to_text = self.fra_token_to_text
                )
                pickle.dump(pickle_data, f)
                
        print(len(self.sentence_list))
        
    def get_eng_dict_size(self):
        return self.eng_token_count + 1
        
    def get_fra_dict_size(self):
        return self.fra_token_count + 1
    
    def get_fra_eos_code(self):
        return self.fra_token_dict['<EOS>']
    
    def get_eng_eos_code(self):
        return self.eng_token_dict['<EOS>']

    def get_fra_start_code(self):
        return self.fra_token_dict['<START>']

    def get_eng_start_code(self):
        return self.eng_token_dict['<START>']

    def get_eng_pad_code(self):
        return self.eng_token_dict['<PAD>']

    def __len__(self):
        return len(self.sentence_list)

    def __getitem__(self, item):
        ret = dict()
        for key in self.sentence_list[item]:
            ret[key] = torch.tensor(self.sentence_list[item][key])
        return ret


def fra_eng_dataset_collate(data):

    MAXMAX_SENTENCE_LEN = 40

    eng_sentences = []
    eng_sentence_lens = []
    fra_sentences = []
    fra_sentence_lens = []
    
    eng_sentences_sorted = []
    eng_sentence_lens_sorted = []
    fra_sentences_sorted = []
    fra_sentence_lens_sorted = []
    
    for s in data:
        sent = s['eng']

        if len(sent) > MAXMAX_SENTENCE_LEN:
            sent = sent[0:MAXMAX_SENTENCE_LEN]
        eng_sentences.append(sent.unsqueeze(dim=1))
        eng_sentence_lens.append(len(sent))

        sent = s['fra']
        if len(sent) > MAXMAX_SENTENCE_LEN:
            sent = sent[0:MAXMAX_SENTENCE_LEN]
        fra_sentences.append(sent.unsqueeze(dim=1))
        fra_sentence_lens.append(len(sent))

    #Rearrange everything by eng sentence lens
    sort_idxes = np.argsort(np.array(eng_sentence_lens))[::-1]
    for idx in sort_idxes:
        eng_sentences_sorted.append(eng_sentences[idx])
        eng_sentence_lens_sorted.append(eng_sentence_lens[idx])
        fra_sentences_sorted.append(fra_sentences[idx])
        fra_sentence_lens_sorted.append(fra_sentence_lens[idx])
    
    return dict(
        eng_sentences = eng_sentences_sorted,
        eng_lens = eng_sentence_lens_sorted,
        fra_sentences = fra_sentences_sorted,
        fra_lens = fra_sentence_lens_sorted
    )
