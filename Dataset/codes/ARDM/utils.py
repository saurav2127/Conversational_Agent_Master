import sys
import os
sys.path.append(os.path.join(os.getcwd(), '../RL'))
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
import numpy as np
from rlutils import generate_n_candidates, convert_sentences_to_strings


def initialize_strategy_count(strategy_dict):
    count_dict_num = {}
    count_dict_str = {}
    strategy_dict_num_to_str = {}
    strategy_dict_num_to_str = {}
    for i in strategy_dict:
        count_dict_num[strategy_dict[i]] = 0
        count_dict_str[i] = 0
        strategy_dict_num_to_str[strategy_dict[i]] = i
    return count_dict_num, count_dict_str, strategy_dict_num_to_str

def get_candidate_lengths(candidate_dict):
    avg_iter_length = []
    for i in candidate_dict:
        for j in candidate_dict[i]:
             avg_iter_length.append(len(j.split()))
    #print(f"Average Candidate Length for {self.num_candidates} candidates generated at each utterance is {np.mean(avg_iter_length)}.")
    return avg_iter_length

def get_num_candidate_with_strategy(candidate_dict, binary_classifier, persuasion_tokenizer, count_dict_num, 
                                    count_dict_str, strategy_dict, strategy_dict_num_to_str):
    pred_labels = []
    for ref in candidate_dict:
        inputs = persuasion_tokenizer(candidate_dict[ref], padding=True, truncation=True)
        output = binary_classifier(inputs['input_ids'], inputs['attention_mask'])
        probs = F.softmax(output.logits, dim=1)
        _, pred_label = torch.topk(probs, k=1, dim=-1)
        pred_labels.extend(pred_label.squeeze(1).tolist())
    num_strategy = np.mean(np.array(pred_labels) != 0)
    for i in pred_labels:
        count_dict_num[i] += 1
    for j in count_dict_num:
        count_dict_str[strategy_dict_num_to_str[j]] = count_dict_num[j]
    return num_strategy, count_dict_num, count_dict_str

class PersuadeDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.tokenizer.max_len = 1500
        self.turn_ending = [628, 198]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        dial_tokens = [self.tokenizer.encode(item) + self.turn_ending for item in self.data[index]]
        # 32 is the encoding for "A:"
        role_ids = [0 if item[0] == 32 else 1 for item in dial_tokens]
        return role_ids, dial_tokens

    def collate(self, unpacked_data):
        return unpacked_data


class SequenceCrossEntropyLoss(nn.Module):
    def __init__(self):
         super().__init__()

    def forward(self, logits, targets, mask, label_smoothing=-1, reduce=None, validate=False):
        return self.sequence_cross_entropy_with_logits(logits, targets, mask, label_smoothing, reduce, validate)

    def sequence_cross_entropy_with_logits(self, logits, targets, mask, label_smoothing, reduce, validate):
        # logits ---> (1, sequences, num_classes)  eg: torch.Size([1, 620, 50257])

        # shape: (batch * sequence_length, num_classes) eg: torch.Size([620, 50257])
        logits_flat = logits.view(-1, logits.size(-1))

        #log_probs = F.log_softmax(logits, dim=-1)
        # shape: (batch * sequence_length, num_classes) eg: torch.Size([620, 50257])
        log_probs_flat = F.log_softmax(logits_flat, dim=-1)

        # targets --> torch.Size([1, 620])
        # shape: (batch * max_len, 1) eg: torch.Size([620, 1])
        targets_flat = targets.view(-1, 1).long()

        if label_smoothing > 0.0:
            num_classes = logits.size(-1)
            smoothing_value = label_smoothing / float(num_classes)
            # Fill all the correct indices with 1 - smoothing value.
            one_hot_targets = torch.zeros_like(log_probs_flat).scatter_(-1, targets_flat, 1.0 - label_smoothing)
            smoothed_targets = one_hot_targets + smoothing_value
            # smoothed_targets eg: torch.Size([620, 50257])
            negative_log_likelihood_flat = -log_probs_flat * smoothed_targets
            negative_log_likelihood_flat = negative_log_likelihood_flat.sum(-1, keepdim=True)
        else:
            # shape: (batch * sequence_length, 1)
            negative_log_likelihood_flat = - torch.gather(log_probs_flat, dim=1, index=targets_flat)

        # shape: (1, batch * sequence_length)
        negative_log_likelihood = negative_log_likelihood_flat.view(-1, logits.shape[1])

        #neg_log_likelihood = -torch.gather(log_probs, -1, targets.unsqueeze(-1))

        # shape : (batch, sequence_length)
        loss = negative_log_likelihood * mask

        if not validate:
            # shape: (batch,)
            loss = loss.sum(1)
        else:
            loss = loss.sum(1) / (mask.sum(1) + 1e-13)
        return loss
