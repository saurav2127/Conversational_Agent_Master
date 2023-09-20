import random
import pdb
import torch
import torch.nn as nn
import numpy as np
import tqdm
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup  #WarmupLinearSchedule
import os
#from PPO import PPOMemory

#os.environ['CUDA_LAUNCH_BLOCKING'] = 1

def convert_sentences_to_strings(sentences:list, tokenizer):
    str_sentences = []
    for i in sentences:
        str_sentences.append(tokenizer.decode(i.tolist()[0][2:-2])) # Excludeqs the zero shot tokens: {A:, B:} and the End of turn tokens: [628, 198]
    return str_sentences

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

def convertDicttoList(data:dict):
    return list(data.values())

def random_split_data(data):
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    train_data = [data[idx] for idx in indices[100:]]
    val_data = [data[idx] for idx in indices[:100]]
    return train_data, val_data

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
        return self.sequence_cross_entropy_with_logits(logits, targets, mask, label_smoothing, reduce)

    def sequence_cross_entropy_with_logits(self, logits, targets, mask, label_smoothing, reduce):
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

@torch.no_grad()
def get_past(dial_inputs, role_ids, model_A, model_B, device):
    past = None
    for turn_num, dial_turn_input in enumerate(dial_inputs):
        if role_ids[turn_num] == 0:
            _, past = model_A(dial_turn_input.to('cuda:4'), past, return_dict=False)
            past = move_past_to_device(past, 'cuda:5')
        else:
            _, past = model_B(dial_turn_input.to('cuda:5'), past, return_dict=False)
            past = move_past_to_device(past, 'cuda:4')
    #past = move_past_to_device(past, device)
    return past

def train_one_iter(batch, update_count, model_A, model_B, num_gradients_accumulation, device, fp16=False):
    role_ids, dialog_tokens = batch
    dial_inputs = [torch.LongTensor(item).unsqueeze(0) for item in dialog_tokens]
    past = None
    losses = 0
    inputs = []
    divide_by = torch.cat(dial_inputs, dim=1)[:, 1:].contiguous().shape[1]
    for turn_num, dial_turn_inputs in enumerate(dial_inputs):
        if role_ids[turn_num] == 0:
            if turn_num == 0:
                past = None
            else:
                past = get_past(inputs, role_ids, model_A, model_B, 'cuda:4')
            dial_turn_inputs = dial_turn_inputs.to('cuda:4')
            logits, _ = model_A(dial_turn_inputs, past_key_values=past, return_dict=False)
            target = dial_turn_inputs[:, 1:].contiguous().to('cuda:4')
            target_mask = torch.ones_like(target).float()
        else:
            past = get_past(inputs, role_ids, model_A, model_B, 'cuda:5')
            dial_turn_inputs = dial_turn_inputs.to('cuda:5')
            logits, _ = model_B(dial_turn_inputs, past_key_values=past, return_dict=False)
            target = dial_turn_inputs[:, 1:].contiguous().to('cuda:5')
            target_mask = torch.ones_like(target).float()
        inputs.append(dial_turn_inputs.cpu())
        loss = criterion(logits[:, :-1].contiguous(), target, target_mask, label_smoothing=0.02, reduce='batch', validate=False)
        loss /= divide_by
        loss /= num_gradients_accumulation
        loss.backward()
        losses += loss.item()
        del past
        del _
        del logits
        del target
        del target_mask
        del dial_turn_inputs

    '''all_logits = torch.cat(all_logits, dim=1)

    all_logits = all_logits[:, :-1].contiguous()
    #target = torch.cat(dial_inputs, dim=1)[:, 1:].contiguous()
    target = torch.cat(target, dim=1)[:, 1:].contiguous()
    target_mask = torch.ones_like(target).float()

    loss = criterion(all_logits, target, target_mask, label_smoothing=-1, reduce='batch') # 0.02 default label_smoothing'''

    #loss /= num_gradients_accumulation
    #loss.backward()

    #record_loss = loss.item() * num_gradients_accumulation
    #perplexity = np.exp(record_loss)
    record_loss = losses
    perplexity = np.exp(losses)
    return record_loss, perplexity

def move_past_to_device(past, device):
    new_past = []
    for i in past: ## here i is a tuple of length 2.
        past_tup = []
        for j in i:
            past_tup.append(j.to(device))
        new_past.append(tuple(past_tup))
    return tuple(new_past)

def validate(dataloader, device):
    with torch.no_grad():
        pbar = progress_bar(dataloader)
        total_ppl = []
        for batch in pbar:
            if sum([len(item) for item in batch[0][1]]) > 1024:
                continue
            role_ids, dialog_tokens = batch[0]
            #dial_inputs = [torch.LongTensor(item).unsqueeze(0).to(device) for item in dialog_tokens]
            dial_inputs = [torch.LongTensor(item).unsqueeze(0) for item in dialog_tokens]
            past = None
            all_logits = []
            target = []
            #pdb.set_trace()
            for turn_num, dial_turn_inputs in enumerate(dial_inputs):
                if role_ids[turn_num] == 0:
                    dial_turn_inputs = dial_turn_inputs.to('cuda:4')
                    if past is not None:
                        past = move_past_to_device(past, 'cuda:4')
                    logits, past = model_A(dial_turn_inputs, past_key_values=past, return_dict=False)
                    all_logits.append(logits.to('cuda:6'))
                    #target.append(dial_turn_inputs.to('cuda:6'))
                else:
                    past = move_past_to_device(past, 'cuda:5')
                    dial_turn_inputs = dial_turn_inputs.to('cuda:5')
                    logits, past = model_B(dial_turn_inputs, past_key_values=past, return_dict=False)
                    all_logits.append(logits.to('cuda:6'))
                    #target.append(dial_turn_inputs.to('cuda:6'))
            all_logits = torch.cat(all_logits, dim=1)
            all_logits = all_logits[:, :-1].contiguous()
            #target = torch.cat(target, dim=1)[:, 1:].contiguous()
            target = torch.cat(dial_inputs, dim=1)[:, 1:].contiguous().to('cuda:6')
            target_mask = torch.ones_like(target).float()
            
            loss = criterion(all_logits, target, target_mask, label_smoothing=-1, reduce='sentence', validate=True)

            ppl = torch.exp(loss)
            total_ppl.extend(ppl.tolist())

        print(f"Epoch {ep} Validation Perplexity: {np.mean(total_ppl)} Variance: {np.var(total_ppl)}")
        return np.mean(total_ppl)

if __name__ == "__main__":

    csvfile = '../dataset/FullData/full_dialog.csv'
    seed() 
    
    loss_per_iteration = []
    loss_per_epoch = []
    avg_epoch_loss = []
    validation_perplexity = []

    print_every = 20
    print_loss = 0
    evaluate_every = 500 # Also saving the model at every evaluation step.

    batch_size = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device = 'cpu'
    num_epochs = 10
    
    data = extract_data(csvfile=csvfile)
    traindata, valdata = random_split_data(data)

    num_gradients_accumulation = 1
    num_train_optimization_steps = num_train_optimization_steps = len(traindata) * num_epochs // batch_size // num_gradients_accumulation

    progress_bar = tqdm.tqdm_notebook

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    traindata = PersuadeDataset(traindata, tokenizer)
    valdata = PersuadeDataset(valdata, tokenizer)


    train_dataloader = DataLoader(dataset=traindata,
                                  shuffle=True,
                                  batch_size=batch_size,
                                  collate_fn = traindata.collate)

    pdb.set_trace()
    for i in next(iter(train_dataloader))[0][1]:
        print(convert_sentences_to_strings([torch.tensor(i).unsqueeze(0)], tokenizer)[0])
    val_dataloader = DataLoader(dataset=valdata,
                                shuffle=False,
                                batch_size=batch_size,
                                collate_fn=valdata.collate)

    model_A = GPT2LMHeadModel.from_pretrained("gpt2-medium")
    model_B = GPT2LMHeadModel.from_pretrained("gpt2-medium")
    #model_A = model_A.to(device)
    model_A = model_A.to('cuda:4')
    model_B = model_B.to('cuda:5')
    #model_B = model_B.to(device)

    criterion = SequenceCrossEntropyLoss()

    param_optimizer = list(model_A.named_parameters()) + list(model_B.named_parameters())
    no_decay = ['bias', 'ln', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=3e-5,
                      eps=1e-06)

    #scheduler = WarmupLinearSchedule(optimizer, warmup_steps=100,
    #                                            t_total=num_train_optimization_steps)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100,  num_training_steps=num_train_optimization_steps)


    update_count = 0
    progress_bar = tqdm.tqdm_notebook
    start = time.time()
    old_ppl = -float('Inf')
    best_ppl = None

    for ep in range(num_epochs):
       
        "Training"
        pbar = progress_bar(train_dataloader)
        model_A.train()
        model_B.train()

        for batch in pbar:
            batch = batch[0]

            # without relative position, we skip dialogs
            if sum([len(item) for item in batch[1]]) > 1024:
                continue

            record_loss, perplexity = train_one_iter(batch, update_count, model_A, model_B, num_gradients_accumulation, device, fp16=False)
            
            print_loss += record_loss
            loss_per_iteration.append(record_loss)
            loss_per_epoch.append(record_loss)

            if (update_count+1) % print_every == 0:
                print(f"Loss: {print_loss/(update_count+1)}")
                print_loss = 0

            update_count += 1

            if update_count % num_gradients_accumulation == num_gradients_accumulation - 1:
                # update for gradient accumulation
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # speed measure
                end = time.time()
                speed = batch_size * num_gradients_accumulation / (end - start)
                start = end

                # show progress
                pbar.set_postfix(loss=record_loss, perplexity=perplexity, speed=speed)
            if update_count >= 1236:        
                model_A.eval()
                model_B.eval()
                ppl = validate(val_dataloader, device)
                if best_ppl is None:
                    best_ppl = ppl
                    torch.save([model_A.state_dict(), model_B.state_dict()], f"medium/models/ARDM_best_model.pth")
                    np.save('count_best_model.npy', np.array([update_count]))
                else:
                    if ppl < best_ppl:
                        best_ppl = ppl
                        torch.save([model_A.state_dict(), model_B.state_dict()], f"medium/models/ARDM_best_model.pth")
                        np.save('count_best_model.npy', np.array([update_count]))
                validation_perplexity.append(ppl)
                perpfile = 'medium/stats_per_iter/val_ppl_' + '.npy'
                np.save(perpfile, validation_perplexity)
                model_A.train()
                model_B.train()
                print('\n')
                print("Best PPL: ", best_ppl)
                print('\n')

            '''if update_count % evaluate_every == 0:
                model_A.eval()
                model_B.eval()
                ppl = validate(val_dataloader, device)
                validation_perplexity.append(ppl)
                # Save Model:
                torch.save([model_A.state_dict(), model_B.state_dict()], f"models/ARDM_{update_count}.pth")'''
                #print('\n')
                #print("Best PPL: ", best_ppl)
                #print('\n')
        
        #perpfile = 'medium/stats_per_iter/val_ppl_' + str(update_count) + '.npy'
        #np.save(perpfile, validation_perplexity)

        avg_epoch_loss.append(np.mean(loss_per_epoch))
        loss_per_epoch = []

        epochlossfile = 'medium/stats_per_iter/train_epoch_loss.npy'
        np.save(epochlossfile, avg_epoch_loss)

    lossiterfile = 'medium/stats_per_iter/train_iter_loss.npy'
    np.save(lossiterfile, loss_per_iteration)

