import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np

def choose_from_top(logits, n=5):
    ind = np.argpartition(logits, -n)[-n:]
    top_prob = logits[ind]

    print(f"top_prob {top_prob}")
    print(f"Sorted {np.sort(logits)[-5:]}")

    top_prob = top_prob / np.sum(top_prob) # Normalize

    choice = np.random.choice(n, 1, p = top_prob)
    token_id = ind[choice][0]
    print(f"top_prob {top_prob} ")
    return token_id

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()

cur_ids = torch.tensor(tokenizer.encode(" The Matrix is everywhere. It is all around us. Even now, in this very room. You can see it when you look out your window or when you turn on your television. You can feel it when you go to work... when you go to church... when you pay your taxes. It is the world that has been pulled over your eyes to blind you from the truth. ")).unsqueeze(0)

with torch.no_grad():

    for i in range(200):

        outputs = model(cur_ids, labels=cur_ids)
        loss, logits = outputs[:2]
        softmax_logits = torch.softmax(logits[0,-1], dim=0) #Take the first(only one) batch and the last predicted embedding
        next_token_id = choose_from_top(softmax_logits.numpy(), n=5) #Randomly(from the given probability distribution) choose the next word from the top n words
        cur_ids = torch.cat([cur_ids, torch.ones((1,1)).long() * next_token_id], dim = 1) # Add the last word

    output_list = list(cur_ids.squeeze().numpy())
    output_text = tokenizer.decode(output_list)
    print(output_text)

