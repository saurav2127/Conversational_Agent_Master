import numpy as np
import random
import pdb
import torch
import tqdm
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
from tqdm.notebook import tqdm_notebook
from utils import SequenceCrossEntropyLoss, PersuadeDataset
#from mediumain import seed, extract_data, random_split_data, PersuadeDataset, SequenceCrossEntropyLoss

def seed(seed=10):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    #torch.cuda.seed()

def extract_data(csvfile):
    df = pd.read_csv(csvfile)
    data = {}
    for i in tqdm.trange(len(df)):
        line = df.iloc[i]
        if line['B2'] not in data:
            data[line['B2']] = []
        if line['B4'] == 0:
            text = "A:" + line['Unit'].strip()
        if line['B4'] == 1:
            text = "B:" + line['Unit'].strip()
        data[line['B2']].append(text)
    data = convertDicttoList(data)
    return data

def random_split_data(data):
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    train_data = [data[idx] for idx in indices[100:]]
    val_data = [data[idx] for idx in indices[:100]]
    return train_data, val_data

def convertDicttoList(data:dict):
    return list(data.values())

def validate(dataloader, device, tokenizer):
    with torch.no_grad():
        total_ppl = []
        pbar = progress_bar(dataloader)
        for batch in pbar:
            if sum([len(item) for item in batch[0][1]]) > 1024:
                continue
            role_ids, dialog_tokens = batch[0]
            dial_inputs = [torch.LongTensor(item).unsqueeze(0).to('cuda:1') for item in dialog_tokens]
            past = None
            all_logits = []
            target = []
            for turn_num, dial_turn_inputs in enumerate(dial_inputs):
                if role_ids[turn_num] == 0:
                    #dial_turn_inputs = dial_turn_inputs.to('cuda')
                    logits, past = model_A(dial_turn_inputs, past_key_values=past, return_dict=False)
                    all_logits.append(logits)
                    target.append(dial_turn_inputs)
                #else:
                #    _, past = model_A(dial_turn_inputs, past_key_values=past, return_dict=False)
            all_logits = torch.cat(all_logits, dim=1)
            all_logits = all_logits[:, :-1].contiguous()
            target = torch.cat(target, dim=1)[:, 1:].contiguous()
            target_mask = torch.ones_like(target).float()

            loss = criterion(all_logits, target, target_mask, label_smoothing=-1, reduce='sentence', validate=True)

            ppl = torch.exp(loss)
            total_ppl.extend(ppl.tolist())
            
        print(f"Validation Perplexity: {np.mean(total_ppl)} Variance: {np.var(total_ppl)}")
        return np.mean(total_ppl) # , np.mean(average_lengths), num_strategy*100

if __name__ == "__main__":

    #csvfile = 'data/full_dialog.csv'
    csvfile = '../RL/dataset/Part2_fully_annotated.csv'
    seed()
    batch_size = 1

    progress_bar = tqdm_notebook

    data = extract_data(csvfile=csvfile)
    traindata, valdata = random_split_data(data)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    valdata = PersuadeDataset(valdata, tokenizer)

    val_dataloader = DataLoader(dataset=valdata,
                                shuffle=False,
                                batch_size=batch_size,
                                collate_fn=valdata.collate)

    model_A = GPT2LMHeadModel.from_pretrained("gpt2-medium")

    #model_A_state_dict, _ = torch.load('medium/ARDM_best_model.pth', map_location = torch.device('cpu'))
    model_A_state_dict = torch.load("../RL/drive/MyDrive/RL_ARDM_medium/Best_Model_ALL_rewards_gridsearch/saved_models/Best_Model_ALL_rewards_gridsearch_5.pth")

    model_A.load_state_dict(model_A_state_dict)
    model_A = model_A.to('cuda:1')

    criterion = SequenceCrossEntropyLoss()

    model_A.eval()

    ppl = validate(val_dataloader,'cpu', tokenizer)

    #np.save('/content/drive/MyDrive/ARDM/medium/stats/Persuader_only_Medium_PPL.npy', np.array(ppl))
