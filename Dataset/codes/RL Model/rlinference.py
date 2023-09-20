import os
import time
import numpy as np
import pandas as pd
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config


def top_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    # Lower temperature makes the model increasingly confident in its top choices,
    # while temperature > 1 decreases confidence. This is done by dividing the before 
    # feeding to the softmax function.

    # Top-k Filtering: Sorting probabilities and zeroing out the probabilities for anything
    # below the k-th token.

    # Top-p Filtering (Nucleus Sampling): Compute the cumulative distribution and cut off as
    # as soon as the CDF exceeds P.

    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        min_values = values[:,-1].unsqueeze(1).repeat(1, logits.shape[-1])
        logits = torch.where(logits < min_values,
                             torch.ones_like(logits, dtype=logits.dtype) * -float('Inf'),
                             logits)

    if top_p > 0.0:
        # Compute probabilities of the sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cum_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        sorted_logits = sorted_logits.masked_fill(sorted_indices_to_remove, filter_value)
        logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)

    return logits

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

model_A = GPT2LMHeadModel.from_pretrained("gpt2-medium")
#model_B = GPT2LMHeadModel.from_pretrained("gpt2-medium")


#device = 'cpu'
device = torch.device("cuda:2")
model_A = model_A.to(device)
#model_B = model_B.to(device)


#model_A_state_dict, model_B_state_dict = torch.load('../ARDM/medium/ARDM_best_model.pth')

model_A_state_dict = torch.load('drive/MyDrive/RL_ARDM_medium/second_originalParams_with_2numcandidates_only_ppl/saved_models/second_originalParams_with_2numcandidates_only_ppl_21.pth')
#model_A_state_dict = torch.load('drive/MyDrive/RL_ARDM_medium/Best_Model_ALL_rewards_gridsearch/saved_models/Best_Model_ALL_rewards_gridsearch_5.pth')

model_A.load_state_dict(model_A_state_dict)
#model_B.load_state_dict(model_B_state_dict)

model_A.eval()
#model_B.eval()


prev_input = tokenizer.encode('A:')
prev_input = torch.LongTensor(prev_input).unsqueeze(0).to(device)

temperature = 0.8
top_k = 400
top_p = 0.9

past = None

sep = [628, 198]

while True:
    sent = []

    with torch.no_grad():
        for i in range(200):
            outputs = model_A(prev_input, past_key_values=past, return_dict=False)
            logits, past = outputs[0], outputs[1]
            logits = logits[:, -1, :] / temperature
            logits = top_filtering(logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(logits, -1)
            prev_input = torch.multinomial(probs, num_samples=1)
            prev_word = prev_input.item()

            if prev_word == 628:
                break
            else:
                sent.append(prev_word)

    print("A:" + tokenizer.decode(sent))
    
    # Finish Tail
    prev_input = torch.LongTensor(sep).unsqueeze(0).to(device)
    outputs = model_A(prev_input, past_key_values=past, return_dict=False)
    past = outputs[1]

    # Input and update B's utterance:
    user = input("B:")

    if user == "quit":
        break

    user = tokenizer.encode("B:" + user)
    prev_input = user + sep
    prev_input = torch.LongTensor(prev_input).unsqueeze(0).to(device)

    #outputs = model_B(prev_input, past_key_values=past, return_dict=True)
    outputs = model_A(prev_input, past_key_values=past, return_dict=True)
    logits, past = outputs[0], outputs[1]

    suffix = tokenizer.encode("A:")
    prev_input = torch.LongTensor(suffix).unsqueeze(0).to(device)


