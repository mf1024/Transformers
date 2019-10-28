import torch
from transformers import *

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')

input_ids = torch.tensor(tokenizer.encode("Joke cracker test.")).unsqueeze(0)  # Batch size 1
outputs = model(input_ids)
last_hidden_states = outputs[0]

print("Yes")
