import sys
sys.path.append('../RL')
from dataset import PersuadeDataset
#from rlutils import jaccard_similarity
import os, pdb
import time
import spacy
import numpy as np
import pandas as pd
import tqdm
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, RobertaForSequenceClassification, RobertaTokenizer
torch.autograd.set_detect_anomaly(True)

def seed(seed=10):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def extract_data(csvfile, classifier, persuasion_classifier):
    df = pd.read_csv(csvfile)
    data = {}
    for i in tqdm.trange(len(df)):
        line = df.iloc[i]
        if line['B2'] not in data:
            data[line['B2']] = []
        if line['B4'] == 0:
            text = "A:" + line['Unit'].strip()
            if classifier and persuasion_classifier:
                #label_id = int(line['label_id'])
                label_id = int(line['encoded_label'])
                persuasion_id = int(line['strategy'])
                tup = (text, label_id, persuasion_id)
            elif classifier and not persuasion_classifier:
                #label_id = int(line['label_id'])
                label_id = int(line['encoded_label'])
                tup = (text, label_id)
            elif persuasion_classifier and not classifier:
                persuasion_id = int(line['strategy'])
                tup = (text, persuasion_id)
            else:
                tup = (text)
        if line['B4'] == 1:
            text = "B:" + line['Unit'].strip()
            if classifier and persuasion_classifier:
                #label_id = int(line['label_id'])
                label_id = None
                persuasion_id = None
                tup = (text, label_id, persuasion_id)
            elif classifier and not persuasion_classifier:
                label_id = None
                tup = (text, label_id)
            elif persuasion_classifier and not classifier:
                persuasion_id = None
                tup = (text, persuasion_id)
            else:
                tup = (text)
        data[line['B2']].append(tup)
    data = convertDicttoList(data)
    return data

def convertDicttoList(data: dict):
    return list(data.values())

def random_split_data(data):
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    train_data = [data[idx] for idx in indices[100:]]
    val_data = [data[idx] for idx in indices[:100]]
    return train_data, val_data

def convert_sentences_to_strings(sentences:list, tokenizer):
    str_sentences = []
    for i in sentences:
        str_sentences.append(tokenizer.decode(i.tolist()[0][2:-2])) # Excludeqs the zero shot tokens: {A:, B:} and the End of turn tokens: [628, 198]
    return str_sentences

def expand_inputs_for_N_candidates(inputs, num_candidates):
    # inputs = inputs[None, ...]
    return inputs.repeat((num_candidates, 1))

def modify_generated_sequence(generated_sequences):
    final_generated_sequences = []
    for i in range(generated_sequences.shape[0]):
        batch_tokens = []
        for j in range(len(generated_sequences[i])):
            if generated_sequences[i][j] != 628 and generated_sequences[i][j] != -1:
                batch_tokens.append(generated_sequences[i][j])
            elif generated_sequences[i][j] == 628:
                batch_tokens.append(generated_sequences[i][j])
                batch_tokens.append(198)
                break
            else:
                break
        final_generated_sequences.append(torch.tensor(batch_tokens).unsqueeze(0))
    return final_generated_sequences

def top_p_candidates(logits, prob=0.92, filter_value=-float('Inf')):
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    cum_sum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    sorted_indices_to_remove = cum_sum > prob
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices_to_remove.scatter_(1, index=sorted_indices, src=sorted_indices_to_remove.clone())
    #indices_to_remove = sorted_indices_to_remove.scatter(1, index=sorted_indices, src=sorted_indices_to_remove)
    logits[indices_to_remove] = filter_value
    return logits

def generate_n_candidates(model, inputs, top_p,  temperature, num_candidates, max_gen_length, past,
                          device, eos_token_id=628, pad_token_id=198):
    curr_len = 2
    inputs = expand_inputs_for_N_candidates(inputs, num_candidates)
    inputs_ = inputs
    generated_sequences = torch.ones((inputs.shape[0], max_gen_length), dtype=torch.long) * -1
    generated_sequences[:, 0:2] = inputs.cpu()
    unfinished_sequences = inputs.new(inputs.shape[0]).fill_(1) #.cpu()
    i = 0
    while True:
        if past:
            if past[0][0].shape[-2] > 1024:
                if not torch.all(generated_sequences==-1):
                    final_generated_sequence, final_generated_log_probs = modify_generated_sequence(generated_sequences, generated_token_log_prob)
                    return final_generated_sequence, final_generated_log_probs, past_to_return
                else:
                    return None, None
        outputs = model(inputs, past, return_dict=False)
        logits, past = outputs[0], outputs[1]
        next_token_logits = logits[:, -1, :].contiguous() / temperature
        if top_p and top_p > 0.0:
            # This returns score after performing softmax function.
            next_token_logits = top_p_candidates(next_token_logits, top_p)
            next_token_log_probs = F.log_softmax(next_token_logits, -1)
            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            #next_token_log_probs = next_token_log_probs.gather(-1, next_tokens)
            next_tokens = next_tokens.squeeze(1)
            if eos_token_id is not None:
                assert pad_token_id is not None # "If eos_token_id is defined, make sure that pad_token_id is defined."
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            generated_sequences[:, curr_len] = next_tokens.cpu()
            inputs = next_tokens.unsqueeze(1).to(device)
            curr_len = curr_len + 1
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())
            if unfinished_sequences.max() == 0:
                break
            if curr_len >= max_gen_length:
                break
    final_generated_sequences = modify_generated_sequence(generated_sequences)
    return final_generated_sequences

def normalize(text, nlp):
    sent = ''
    doc = nlp(text)
    for token in doc:
        if not token.is_punct:
            sent += token.lemma_
            sent += ' '
    return sent

def jaccard_similarity(context_sentence_list, generated_sentence, nlp):
    str1 = context_sentence_list[0]
    str1 = normalize(str1, nlp)
    str1 = set(str1.split())
    jacc_score = []
    for i in generated_sentence:
        str2 = i
        str2 = normalize(str2, nlp)
        str2 = set(str2.split())
        sim_score = float(len(str1 & str2)) / len(str1 | str2)
        jacc_score.append(sim_score)
    return jacc_score

def filter_response(generated_responses, jaccard_score):
    filtered_responses = []
    for i in range(len(generated_responses)):
        if jaccard_score[i] >= 0.5:
            continue
        else:
            filtered_responses.append(generated_responses[i])
    return filtered_responses

def response_with_strategy(binary_classifier, binary_tokenizer, generated_responses, device):
    inputs = binary_tokenizer(generated_responses, padding=True, return_tensors='pt')
    outputs = binary_classifier(inputs['input_ids'].to(device), inputs['attention_mask'].to(device))
    probs = F.softmax(outputs.logits, dim=-1)
    _, pred_label = torch.topk(probs, k=1)
    try:
        index_containing_strategy = list(np.where(np.array(pred_label.cpu()) != 0)[0])
    except:
        pass
    if len(index_containing_strategy) > 1:
        # Select the one with bigger length:
        lengths = [len(generated_responses[i].split()) for i in index_containing_strategy]
        return generated_responses[np.argmax(lengths)]
        ## Randomly select from the ones containing strategy.
        #randomly_selected_index = np.random.choice(index_containing_strategy)
        #return generated_responses[randomly_selected_index]
    elif len(index_containing_strategy) == 1:
        return generated_responses[index_containing_strategy[0]]
    else:
        lengths = [len(i.split()) for i in generated_responses]
        return generated_responses[np.argmax(lengths)]
        #return np.random.choice(generated_responses)

def get_best_candidate(generated_sequences, binary_classifier, tokenizer, binary_tokenizer, device, context_sentence_list, nlp):
    generated_responses = convert_sentences_to_strings(generated_sequences, tokenizer)
    if len(context_sentence_list) != 0:
        jaccard_score = jaccard_similarity(context_sentence_list, generated_responses, nlp)
        filtered_responses = filter_response(generated_responses, jaccard_score)
    else:
        filtered_responses = generated_responses
    best_response = response_with_strategy(binary_classifier, binary_tokenizer, filtered_responses, device)
    return best_response


csvfile = '../RL/dataset/Part2_fully_annotated.csv'
seed()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
batch_size = 1

#progress_bar = tqdm_notebook

data = extract_data(csvfile=csvfile, classifier=False, persuasion_classifier=False)
traindata, valdata = random_split_data(data)
valdata = PersuadeDataset(valdata, tokenizer)
val_dataloader = DataLoader(dataset=valdata,
                            shuffle=True,
                            batch_size=batch_size,
                            collate_fn=valdata.collate)

#tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

nlp = spacy.load("en_core_web_sm")

ARDM_model_A = GPT2LMHeadModel.from_pretrained("gpt2-medium")
RL_model_A = GPT2LMHeadModel.from_pretrained("gpt2-medium")
#model_B = GPT2LMHeadModel.from_pretrained("gpt2-medium")

device = 'cuda' #torch.device("cuda")

RL_model_A = RL_model_A.to(device)
ARDM_model_A = ARDM_model_A.to(device)
#model_B = model_B.to(device)

ARDM_model_A_state_dict, _ = torch.load('medium/ARDM_best_model.pth')
RL_model_A_state_dict = torch.load('../RL/drive/MyDrive/RL_ARDM_medium/Best_Model_ALL_rewards_gridsearch/saved_models/Best_Model_ALL_rewards_gridsearch_5.pth')

RL_model_A.load_state_dict(RL_model_A_state_dict)
ARDM_model_A.load_state_dict(ARDM_model_A_state_dict)

RL_model_A.eval()
ARDM_model_A.eval()

## Loading the Binary Classifier for selecting the best out of the generated candidates:
per_classifier_filename = '../persuasion_classifier/models/roberta/32/lowest_eval_model.pt'
persuasion_num_labels = 2 ## Binary classifier
model_dict = torch.load(per_classifier_filename)
binary_classifier = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=persuasion_num_labels)
binary_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
binary_classifier.config.problem_type = 'single_label_classification'
binary_classifier.load_state_dict(model_dict['state_dict'])
binary_classifier = binary_classifier.to(device)
binary_classifier.eval()


prev_input = tokenizer.encode('A:')
prev_input = torch.LongTensor(prev_input).unsqueeze(0).to(device)

temperature = 0.8
top_k = 400
top_p = 0.9
max_length = 100
num_candidates = 3
#device = 'cuda:1'

RL_past = None
ARDM_past = None

sep = [628, 198]


ground_truth = []
RL_candidates = []
ARDM_candidates = []
B4 = []
ID = []

id_ = 0
count = 0

with torch.no_grad():
    for idx, batch in enumerate(val_dataloader):
        RL_past = None
        ARDM_past = None
        if sum([len(item) for item in batch[0][1]]) > 1024:
            continue
        role_ids, dialog_tokens = batch[0]
        context_sentence_list = []
        #B4.extend(role_ids)
        dial_inputs = [torch.LongTensor(item).unsqueeze(0) for item in dialog_tokens]
        for num_turn, dialog_turn_inputs in enumerate(dial_inputs):
            assert not np.any(np.isnan(dialog_turn_inputs).cpu().numpy()), 'Inputs Dialog contains Nan value.'
            dialog_turn_inputs = dialog_turn_inputs.to(device)
            if role_ids[num_turn] == 0:
                generated_sequences = generate_n_candidates(RL_model_A, torch.tensor(tokenizer.encode("A:")).unsqueeze(0).to(device),
                                                               top_p=top_p,  temperature=temperature,
                                                               num_candidates=num_candidates, max_gen_length=max_length, past=RL_past,
                                                               device=device, eos_token_id=628, pad_token_id=198)

                RL_best_candidate = get_best_candidate(generated_sequences, binary_classifier, tokenizer, binary_tokenizer, device, context_sentence_list, nlp)
                #RL_candidate_string = convert_sentences_to_strings(RL_best_candidate, tokenizer)
                RL_candidates.append(RL_best_candidate)
                _, RL_past = RL_model_A(expand_inputs_for_N_candidates(dialog_turn_inputs, num_candidates), RL_past, return_dict=False)

                generated_sequences = generate_n_candidates(ARDM_model_A, torch.tensor(tokenizer.encode("A:")).unsqueeze(0).to(device),
                                                               top_p=top_p,  temperature=temperature,
                                                               num_candidates=1, max_gen_length=max_length, past=ARDM_past,
                                                               device=device, eos_token_id=628, pad_token_id=198)
                ARDM_candidate_string = convert_sentences_to_strings(generated_sequences, tokenizer)[0]
                ARDM_candidates.append(ARDM_candidate_string)
                _, ARDM_past = ARDM_model_A(dialog_turn_inputs, ARDM_past, return_dict=False)
                
                ground_truth_string = convert_sentences_to_strings([dialog_turn_inputs], tokenizer)[0]
                ground_truth.append(ground_truth_string)
                B4.append(0)
                ID.append(id_)
                #print(f"Ground truth: {ground_truth_string}, generated: {RL_best_candidate,  ARDM_candidate_string}")
                if len(context_sentence_list) == 0:
                    context_sentence_list.append(ground_truth_string)
                else:
                    context_sentence_list.pop()
                    context_sentence_list.append(ground_truth_string)
            else:
                _, RL_past = RL_model_A(expand_inputs_for_N_candidates(dialog_turn_inputs, num_candidates), RL_past, return_dict=False)
                _, ARDM_past = ARDM_model_A(dialog_turn_inputs, ARDM_past, return_dict=False)
                RL_candidates.append(None)
                ARDM_candidates.append(None)
                ground_truth_string = convert_sentences_to_strings([dialog_turn_inputs], tokenizer)[0]
                ground_truth.append(ground_truth_string)
                B4.append(1)
                ID.append(id_)
            count +=1
        id_ += 1
        if count >= 300:
            break

df = pd.DataFrame({'ID': ID,
                   'B4': B4,
                   'ground_truth': ground_truth,
                   'RL_candidate': RL_candidates,
                   'ARDM_candidate': ARDM_candidates})

df.head()

df.to_csv('human_evaluation/Final_evaluation_improved.csv')
