import numpy as np
import torch.nn as nn
from nltk.translate.bleu_score import sentence_bleu as sentblu
from nltk.translate.meteor_score import meteor_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import functools
import operator
import os
import pdb
import spacy
import pandas as pd
import json
import tqdm
import datetime
from tqdm.notebook import tqdm_notebook
import random
import pdb
from rlutils import collect_samples, ppo_step, generate_n_candidates, convert_sentences_to_strings, expand_inputs_for_N_candidates
from torch.utils.data import DataLoader, Dataset
from loss import SequenceCrossEntropyLoss
from ppo import PPOMemory
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup, RobertaForSequenceClassification, RobertaTokenizer
from simpletransformers.classification import ClassificationModel, ClassificationArgs
torch.cuda.empty_cache()
torch.autograd.set_detect_anomaly(True)
from dataset import PersuadeDataset

class Trainer():
    def __init__(self, modelname, csvfile, n_epochs, print_every, learning_rate, epsilon, human_reward, average_sent_loss, use_bleu, use_meteor, device, beta2,
                 num_candidates, max_candidate_length, beta, top_p, warmup_steps, pad_token_id, evaluate_every, use_jaccard, use_cosine, use_emo_classifier, per_num_labels,
                 mini_batch, temperature, use_recent_past, recompute_log_prob, use_persuasion_classifier, per_classifier_filename, bin_classifier_filename, emo_num_labels,
                 gamma1, gamma2, gamma3, gamma4, train_single_model=False, single_model_to_train=None, loadModel=False, batch_size=None,  loadFilename=None,
                 emo_classifier_filename=None, seedvalue=10):

        self.seedvalue = seedvalue
        self.train_single_model = train_single_model
        self.single_model_to_train = single_model_to_train
        self.nlp = spacy.load("en_core_web_sm")
        self.human_reward = human_reward
        self.seed(seedvalue)
        #set_random_seed(seedvalue)
        self.use_recent_past = use_recent_past
        self.temperature=temperature
        self.use_jacc = use_jaccard
        self.use_cosine = use_cosine
        self.use_bleu = use_bleu
        self.use_meteor = use_meteor
        ## user can only use Either Jaccard or Cosine Similarity:
        if use_cosine and use_jaccard:
            raise ValueError('Both Cosine and Jaccard Similarity cannot be True. Choose Either one as reward.')
        elif not use_cosine and not use_jaccard:
            raise ValueError('Both Cosine and Jaccard Similarity cannot be False. Select either Cosine or Jaccard Similarity.')
        self.average_sent_loss = average_sent_loss
        self.mini_batch = mini_batch
        self.evaluate_every = evaluate_every
        self.csvfile = csvfile
        self.modelname = modelname
        self.n_epochs = n_epochs
        self.print_every = print_every
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.recompute_log_prob = recompute_log_prob
        self.num_candidates = num_candidates
        self.pad_token_id = pad_token_id
        self.max_candidate_length = max_candidate_length
        self.beta = beta
        self.beta2  = beta2
        self.top_p = top_p
        self.warmup_steps = warmup_steps
        self.batch_size = batch_size
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.use_emo_classifier = use_emo_classifier
        self.num_labels = per_num_labels
        self.num_emo_labels = emo_num_labels
        #self.device = torch.device('cpu')
        if self.use_emo_classifier and emo_classifier_filename:
            self.classifier = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=self.num_emo_labels)
            self.classifier.config.problem_type = 'single_label_classification'
            model_state_dict = torch.load(emo_classifier_filename)
            self.classifier.load_state_dict(model_state_dict['state_dict'])
            self.classifier.eval()
            self.classifier = self.classifier.to(self.device)
            #class_args = ClassificationArgs()
            #class_args.silent = True
            #self.classifier = ClassificationModel("roberta", emo_classifier_filename, args=class_args, use_cuda=True)
            print('Emotional Classifier Loaded! (in Evaluation Mode)')
        elif self.use_emo_classifier and not emo_classifier_filename:
            raise ValueError('Emotional classifier use set to True, but filename to load from not defined.')
        else:
            self.classifier= None
            print('Not Using Emotional Classifier')
        self.loadModel = loadModel
        self.loadFilename = loadFilename
        self.make_model_save_dir()
        self.make_stats_dir()
        self.use_persuasion_classifier = use_persuasion_classifier
        if self.use_persuasion_classifier and per_classifier_filename:
            model_dict = torch.load(per_classifier_filename)
            self.persuasion_classifier = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=self.num_labels)
            self.persuasion_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
            self.persuasion_classifier.config.problem_type = 'single_label_classification'
            self.persuasion_classifier.load_state_dict(model_dict['state_dict'])
            self.persuasion_classifier = self.persuasion_classifier.to(self.device)
            self.persuasion_classifier.eval()
            print('Persuasion Classifier Loaded! (in Evaluation Mode)')
            # Loading the binary classifier to recognize utterances with strategies
            '''self.binary_classifier = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=2)
            self.binary_classifier.config.problem_type = 'single_label_classification'
            model_dict = torch.load(bin_classifier_filename)
            self.binary_classifier.load_state_dict(model_dict['state_dict'])
            self.binary_classifier = self.binary_classifier.to(self.device)
            self.binary_classifier.eval()
            print('Binary Classifier Loaded! (in Evaluation Mode)')'''
            self.binary_classifier = None
        elif self.use_persuasion_classifier and not per_classifier_filename:
            raise ValueError('Persuasion classifier use set to True, but filename to load from not defined.')
        else:
            self.persuasion_classifier = None
            #self.persuasion_tokenizer = None
            self.persuasion_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.getDataset()
        self.initialize_models()
        self.configure_optimizer()
        self.buffer_memory = PPOMemory()
        self.saveModelConfig()
        self.criterion = SequenceCrossEntropyLoss()
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3
        self.gamma4 = gamma4
        self.per_classifier_filename = per_classifier_filename
        self.emo_classifier_filename = emo_classifier_filename
        '''self.strategy_dict = label_dict = {'credibility-appeal': 2,
                                           'donation-information': 6,
                                           'emotion-appeal': 8,
                                           'foot-in-the-door': 7,
                                           'logical-appeal': 3,
                                           'no_strategy': 0,
                                           'personal-related-inquiry': 4,
                                           'personal-story': 10,
                                           'self-modeling': 9,
                                           'source-related-inquiry': 5,
                                           'task-related-inquiry': 1}
        self.initialize_strategy_count()'''

    def initialize_classifier_models(self):
        self.classifier = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=self.num_emo_labels)
        self.classifier.config.problem_type = 'single_label_classification'
        model_state_dict = torch.load(self.emo_classifier_filename)
        self.classifier.load_state_dict(model_state_dict['state_dict'])
        self.classifier.eval()
        self.classifier = self.classifier.to(self.device)
        #class_args = ClassificationArgs()
        #class_args.silent = True
        #self.classifier = ClassificationModel("roberta", emo_classifier_filename, args=class_args, use_cuda=True)
        print('Emotional Classifier Loaded! (in Evaluation Mode)')

        model_dict = torch.load(self.per_classifier_filename)
        self.persuasion_classifier = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=self.num_labels)
        self.persuasion_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.persuasion_classifier.config.problem_type = 'single_label_classification'
        self.persuasion_classifier.load_state_dict(model_dict['state_dict'])
        self.persuasion_classifier = self.persuasion_classifier.to(self.device)
        self.persuasion_classifier.eval()
        print('Persuasion Classifier Loaded! (in Evaluation Mode)')


    def saveModelConfig(self):
        if self.train_single_model:
            config_model_train = self.single_model_to_train
            print('Training Only :', self.single_model_to_train)
        else:
            config_model_train = 'Both Models being Trained.'
            print('Both Models being Trained.')
        config = {'Basic Info': [datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S")],
                  'NOTES': 'GPT2-MEDIUM',
                  'modelname': self.modelname,
                  'Training only one Model': self.train_single_model,
                  'Training Models': config_model_train,
                  'emotional_classifier': self.use_emo_classifier,
                  'persuasion_classifier': self.use_persuasion_classifier,
                  'num_labels_persuasion': self.num_labels,
                  'use_meteor':self.use_meteor,
                  'beta2': self.beta2,
                  'device': self.device,
                  'use_jaccard_similarity': self.use_jacc,
                  'use_bleu': self.use_bleu,
                  'modelLoaded': self.loadFilename,
                  'human_reward': self.human_reward,
                  'average_sent_loss' : self.average_sent_loss,
                  'n_epochs': self.n_epochs,
                  'use_recent_past': self.use_recent_past,
                  'temperature': self.temperature,
                  'learning_rate': self.learning_rate,
                  'epsilon': self.epsilon,
                  'num_candidates': self.num_candidates,
                  'pad_token_id': self.pad_token_id,
                  'max_candidate_length': self.max_candidate_length,
                  'recompute_log_prob': self.recompute_log_prob,
                  'beta': self.beta,
                  'evaluate_every': self.evaluate_every,
                  'top_p': self.top_p,
                  'warmup_steps': self.warmup_steps,
                  'batch_size':self.batch_size,
                  'seed': self.seedvalue}
        configfilename = os.path.join(self.savefolder, self.modelname, 'config')
        if not os.path.exists(configfilename):
            os.makedirs(configfilename)
        configfilename = configfilename + '/config' + '_' + self.modelname + '.json'
        with open(configfilename, 'w') as f:
            json.dump(config, f)

    def seed(self,seed=10):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    '''def seed(self, seed=10):
        random.seed(seed:
        np.random.seed(seed)
        torch.manual_seed(_hashed_seed)
        torch.cuda.manual_seed(_hashed_seed)
        torch.cuda.manual_seed_all(_hashed_seed)'''

    def extract_data(self, csvfile):
        df = pd.read_csv(csvfile)
        data = {}
        for i in tqdm.trange(len(df)):
            line = df.iloc[i]
            if line['B2'] not in data:
                data[line['B2']] = []
            if line['B4'] == 0:
                text = "A:" + line['Unit'].strip()
                if self.classifier and self.persuasion_classifier:
                    #label_id = int(line['label_id'])
                    label_id = int(line['encoded_label'])
                    persuasion_id = int(line['strategy'])
                    tup = (text, label_id, persuasion_id)
                elif self.classifier and not self.persuasion_classifier:
                    #label_id = int(line['label_id'])
                    label_id = int(line['encoded_label'])
                    tup = (text, label_id)
                elif self.persuasion_classifier and not self.classifier:
                    persuasion_id = int(line['strategy'])
                    tup = (text, persuasion_id)
                else:
                    tup = (text)
            if line['B4'] == 1:
                text = "B:" + line['Unit'].strip()
                if self.classifier and self.persuasion_classifier:
                    #label_id = int(line['label_id'])
                    label_id = None
                    persuasion_id = None
                    tup = (text, label_id, persuasion_id)
                elif self.classifier and not self.persuasion_classifier:
                    label_id = None
                    tup = (text, label_id)
                elif self.persuasion_classifier and not self.classifier:
                    persuasion_id = None
                    tup = (text, persuasion_id)
                else:
                    tup = (text)
            data[line['B2']].append(tup)
        data = self.convertDicttoList(data)
        return data

    def convertDicttoList(self, data: dict):
        return list(data.values())

    def random_split_data(self, data):
        indices = np.arange(len(data))
        np.random.shuffle(indices)
        train_data = [data[idx] for idx in indices[100:]]
        val_data = [data[idx] for idx in indices[:100]]
        return train_data, val_data

    def getDataset(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        data = self.extract_data(self.csvfile)
        self.traindata, self.valdata = self.random_split_data(data)
        if self.classifier and self.persuasion_classifier:
            use_emotion_labels=True
            use_persuasion_labels=True
        elif self.classifier and not self.persuasion_classifier:
            use_emotion_labels=True
            use_persuasion_labels=False
        elif not self.classifier and self.persuasion_classifier:
            use_emotion_labels=False
            use_persuasion_labels=True
        else:
            use_emotion_labels=False
            use_persuasion_labels=False
        traindata_ = PersuadeDataset(self.traindata, self.tokenizer,
                                     use_emotion_labels=use_emotion_labels,
                                     use_persuasion_labels=use_persuasion_labels)
        self.turn_ending = traindata_.get_turn_ending()
        valdata_ = PersuadeDataset(self.valdata, self.tokenizer,
                                   use_emotion_labels=use_emotion_labels,
                                   use_persuasion_labels=use_persuasion_labels)
        self.train_dataloader = DataLoader(dataset=traindata_,
                                           shuffle=True,
                                           batch_size=self.batch_size,
                                           collate_fn=traindata_.collate)
        self.val_dataloader = DataLoader(dataset=valdata_,
                                         shuffle=False,
                                         batch_size=self.batch_size,
                                         collate_fn=valdata_.collate)

    def initialize_models(self):
        if not self.train_single_model:
            self.model_A = GPT2LMHeadModel.from_pretrained("gpt2-medium")
            self.model_B = GPT2LMHeadModel.from_pretrained("gpt2-medium")
            self.model_A_ref = GPT2LMHeadModel.from_pretrained("gpt2-medium")
            self.model_B_ref = GPT2LMHeadModel.from_pretrained("gpt2-medium")
        else:
            if self.single_model_to_train == 'persuader':
                self.model_A = GPT2LMHeadModel.from_pretrained("gpt2-medium")
                self.model_A_ref = GPT2LMHeadModel.from_pretrained("gpt2-medium")
                #self.model_B = GPT2LMHeadModel.from_pretrained("gpt2-medium")
            else:
                self._model_B = GPT2LMHeadModel.from_pretrained("gpt2-medium")
                self.model_B_ref = GPT2LMHeadModel.from_pretrained("gpt2-medium")

        if self.loadModel:
            if self.loadFilename:
                model_A_state_dict, model_B_state_dict = torch.load(self.loadFilename)#, map_location=self.device)
                if not self.train_single_model:
                    self.model_A.load_state_dict(model_A_state_dict)
                    self.model_A_ref.load_state_dict(model_A_state_dict)
                    self.model_B.load_state_dict(model_B_state_dict)
                    self.model_B_ref.load_state_dict(model_B_state_dict)
                    self.model_A = self.model_A.to(self.device)
                    self.model_A_ref = self.model_A_ref.to(self.device)
                    self.model_B = self.model_B.to(self.device)
                    self.model_B_ref = self.model_B_ref.to(self.device)
                    self.model_A.train()
                    self.model_B.train()
                    self.model_A_ref.eval()
                    self.model_B_ref.eval()
                else:
                    if self.single_model_to_train == 'persuader':
                        self.model_A.load_state_dict(model_A_state_dict)
                        self.model_A_ref.load_state_dict(model_A_state_dict)
                        self.model_A = self.model_A.to(self.device)
                        self.model_A_ref = self.model_A_ref.to(self.device)
                        self.model_A.train()
                        self.model_A_ref.eval()
                        #self.model_B.load_state_dict(model_B_state_dict) 
                        #self.model_B = self.model_B.to('cuda')
                        #self.model_B.eval()
                        self.model_B = None
                        self.model_B_ref = None
                    else:
                        self.model_B.load_state_dict(model_B_state_dict)
                        self.model_B_ref.load_state_dict(model_B_state_dict)
                        self.model_B = self.model_B.to(self.device)
                        self.model_B_ref = self.model_B_ref.to(self.device)
                        self.model_B.train()
                        self.model_B_ref.eval()
                        self.model_A = None
                        self.model_A_ref = None
                print('\n')
                print("Models loaded from file ", self.loadFilename)
            else:
                print('Models not loaded since directory not provided.')
        print(f"Models Initalized!")
        print('\n')

    def configure_optimizer(self):
        self.num_train_optimization_steps = self.n_epochs * len(self.traindata) // self.batch_size
        #self.num_train_optimization_steps = len(self.traindata)
        if not self.train_single_model:
            param_optimizer = list(self.model_A.named_parameters()) + list(self.model_B.named_parameters())
        else:
            if self.single_model_to_train == 'persuader':
                param_optimizer = list(self.model_A.named_parameters())
        no_decay = ['bias', 'ln', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = optimizer = AdamW(optimizer_grouped_parameters,
                                           lr=self.learning_rate,
                                           eps=1e-06)

        #self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
        #                                                 num_warmup_steps=self.warmup_steps,
        #                                                 num_training_steps=self.num_train_optimization_steps)

        '''self.scheduler = WarmupLinearSchedule(self.optimizer,
                                              warmup_steps=self.warmup_steps,
                                              t_total=self.num_train_optimization_steps)'''
    def initialize_strategy_count(self):
        self.count_dict_num = {}
        self.count_dict_str = {}
        self.strategy_dict_num_to_str = {}
        for i in self.strategy_dict:
            self.count_dict_num[self.strategy_dict[i]] = 0
            self.count_dict_str[i] = 0
            self.strategy_dict_num_to_str[self.strategy_dict[i]] = i

    def get_candidate_lengths(self, candidate_dict):
        avg_iter_length = []
        for i in candidate_dict:
            for j in candidate_dict[i]:
                 avg_iter_length.append(len(j.split()))
        #print(f"Average Candidate Length for {self.num_candidates} candidates generated at each utterance is {np.mean(avg_iter_length)}.")
        return avg_iter_length

    def get_num_candidate_with_strategy(self, candidate_dict):
        pred_labels = []
        for ref in candidate_dict:
            inputs = self.persuasion_tokenizer(candidate_dict[ref], padding=True, truncation=True, return_tensors='pt')
            output = self.binary_classifier(inputs['input_ids'].to(self.device), inputs['attention_mask'].to(self.device))
            probs = F.softmax(output.logits, dim=1)
            _, pred_label = torch.topk(probs, k=1, dim=-1)
            pred_labels.extend(pred_label.squeeze(1).tolist())
        num_strategy = np.mean(np.array(pred_labels) != 0)
        for i in pred_labels:
            self.count_dict_num[i] += 1
        for j in self.count_dict_num:
            self.count_dict_str[self.strategy_dict_num_to_str[j]] = self.count_dict_num[j]
        return num_strategy

    def validate_model(self, dataloader):
        with torch.no_grad():
            if not self.train_single_model:
                self.model_A.eval()
                self.model_B.eval()
            else:
                if self.single_model_to_train == 'persuader':
                    self.model_A.eval()
                else:
                    self.model_B.eval()

            with torch.no_grad():
                progress_bar = tqdm_notebook
                pbar = progress_bar(dataloader)
                total_ppl = []
                total_loss = []
                candidates_dict = {}
                #pdb.set_trace()
                for batch in pbar:
                    if sum([len(item) for item in batch[0][1]]) > 1024:
                        continue
                    if not self.persuasion_classifier and self.classifier:
                        role_ids, dialog_tokens, emotion_label = batch[0]
                    elif self.persuasion_classifier and self.classifier:
                        role_ids, dialog_tokens, emotion_labels, persuasion_label = batch[0]
                    elif self.persuasion_classifier and not self.classifier:
                        role_ids, dialog_tokens, persuasion_label = batch[0]
                    else:
                        role_ids, dialog_tokens = batch[0]
                    #dial_inputs = [torch.LongTensor(item).unsqueeze(0).to(self.device) for item in dialog_tokens]
                    dial_inputs = [torch.LongTensor(item).unsqueeze(0) for item in dialog_tokens]
                    past = None
                    past_ = None
                    all_logits = []
                    target = []

                    for turn_num, dial_turn_inputs in enumerate(dial_inputs):
                        if not self.train_single_model:
                            if role_ids[turn_num] == 0:
                                outputs = self.model_A(dial_turn_inputs, past_key_values=past, return_dict=False)
                                past = outputs[1]
                                all_logits.append(outputs[0])
                            else:
                                outputs = self.model_B(dial_turn_inputs, past_key_values=past, return_dict=False)
                                past = outputs[1]
                                all_logits.append(outputs[0])
                        else:
                            if self.single_model_to_train == 'persuader':
                                if role_ids[turn_num] == 0:
                                    #dial_turn_str = convert_sentences_to_strings([dial_turn_inputs], self.tokenizer)[0]
                                    dial_turn_inputs = dial_turn_inputs.to(self.device)
                                    outputs = self.model_A(dial_turn_inputs, past_key_values=past, return_dict=False)
                                    past = outputs[1]
                                    all_logits.append(outputs[0])
                                    target.append(dial_turn_inputs)
                                    '''generated_sequence, generated_log_probs  = generate_n_candidates(self.model_A,
                                                                                          torch.tensor(self.tokenizer.encode("A:")).unsqueeze(0).to('cuda'),
                                                                                          self.top_p,
                                                                                          eos_token_id=self.turn_ending[0],
                                                                                          pad_token_id=self.turn_ending[1],
                                                                                          num_candidates=self.num_candidates,
                                                                                          max_gen_length=200,
                                                                                          temperature=self.temperature,
                                                                                          past=past_,
                                                                                          device=self.device)
                                    output = self.model_A(expand_inputs_for_N_candidates(dial_turn_inputs, self.num_candidates), past_, return_dict=False)
                                    past_ = output[1]
                                    candidates_dict[dial_turn_str] = convert_sentences_to_strings(generated_sequence, self.tokenizer)'''
                    all_logits = torch.cat(all_logits, dim=1)
                    all_logits = all_logits[:, :-1].contiguous()

                    if not self.train_single_model:
                        target = torch.cat(dial_inputs, dim=1)[:, 1:].contiguous()
                    else:
                        target = torch.cat(target, dim=1)[:, 1:].contiguous()
                    
                    target_mask = torch.ones_like(target).float()

                    loss = self.criterion(all_logits, target, target_mask, label_smoothing=-1, reduce='sentence')
                    total_loss.extend(loss.tolist())

                    ppl = torch.exp(loss)
                    total_ppl.extend(ppl.tolist())
                    
                print('\n')
                print(f"Validation Perplexity: {np.mean(total_ppl)}")

                '''average_lengths = self.get_candidate_lengths(candidates_dict)
                num_strategy = self.get_num_candidate_with_strategy(candidates_dict)
                print(f"Percenatge of candidates with strategy: {num_strategy*100.}%")
                print(f"Average candidate length: {np.mean(average_lengths)}")

        return np.mean(total_ppl), np.mean(total_loss), np.mean(average_lengths), num_strategy*100'''
        return np.mean(total_ppl), np.mean(total_loss)
    
    def make_stats_dir(self):
        self.statsfolder = os.path.join(os.getcwd(), self.savefolder, self.modelname, 'stats')
        if not os.path.exists(self.statsfolder):
            os.makedirs(self.statsfolder)

    def make_model_save_dir(self):
        self.savefolder = os.path.join(os.getcwd(), 'drive/MyDrive/RL_ARDM_medium')
        if not os.path.exists(self.savefolder):
            print("Model save folder doesn't exist.")
            os.makedirs(self.savefolder)
            print(f"Created folder {self.savefolder} to save the models.")

    def save_models(self, num_iter):
        modeldir = os.path.join(self.savefolder, self.modelname, 'saved_models')
        if not os.path.exists(modeldir):
            os.makedirs(modeldir)
            print('Created Directory for saving models!')
        filename = modeldir + '/' + self.modelname + '_' + str(num_iter) + ".pth"
        #torch.save([self.model_A.state_dict(), self.model_B.state_dict()], filename)
        torch.save(self.model_A.state_dict(), filename)

    def modified_train_one_iter(self, batch):
        dial_inputs, role_ids, scores_dict = collect_samples(batch, model_A=self.model_A_ref, model_B=self.model_B, top_p=self.top_p,
                        eos_token_id=self.turn_ending[0], pad_token_id=self.turn_ending[1], average_sent_loss=self.average_sent_loss, 
                        max_gen_length=self.max_candidate_length, buffer_memory=self.buffer_memory, use_bleu=self.use_bleu, use_meteor=self.use_meteor, beta=self.beta,
                        device=self.device, num_candidates=self.num_candidates, human_reward=self.human_reward, use_jacc=self.use_jacc, persuasion_tokenizer=self.persuasion_tokenizer, 
                        persuasion_classifier=self.persuasion_classifier, beta2=self.beta2, tokenizer=self.tokenizer, temperature=self.temperature,
                        use_recent_past=self.use_recent_past, classifier=self.classifier, recompute_log_prob=self.recompute_log_prob,
                        nlp=self.nlp, train_single_model=self.train_single_model, model_to_train=self.single_model_to_train, gamma1=self.gamma1, gamma2=self.gamma2,
                        gamma3=self.gamma3, gamma4=self.gamma4)

        log_dict = ppo_step(model_A=self.model_A, model_B=self.model_B, buffer_memory=self.buffer_memory, train_single_model=self.train_single_model, dial_inputs= dial_inputs,
                            model_to_train=self.single_model_to_train,  device=self.device, ppo_epsilon=self.epsilon, beta=self.beta, num_candidates=self.num_candidates,
                            use_recent_past=self.use_recent_past, average_sent_loss=self.average_sent_loss, optimizer=self.optimizer, role_ids=role_ids)
        self.buffer_memory.clear_memory()
        return log_dict, scores_dict 
 
    def train(self):
        update_count = 0
        progress_bar = tqdm_notebook

        val_ppl = []
        val_loss = []

        rewards = []
        kl = []
        clip_frac = []

        meteor_scores = []
        emotion_scores = []
        jacc_scores = []
        persuasion_scores = []
        emotion_actual_probs = []
        emotion_other_probs = []
        per_actual_probs = []
        per_other_probs = []
        
        #candidate_lengths = []
        #percent_candidates_with_strategy = []

        best_ppl = None
        #length = None
        iters = None
        #strategies = None

        pbar = progress_bar(self.train_dataloader)

        for i in range(self.n_epochs):
            if not self.train_single_model:
                self.model_A.train()
                self.model_B.train()
            else:
                if self.single_model_to_train == 'persuader':
                    self.model_A.train()
            #pdb.set_trace()
            for batch in pbar:
                if sum([len(item) for item in batch[0][1]]) > 1024 - self.max_candidate_length:
                    continue

                print(f"ITERATION: {update_count}")

                batch = batch[0]
                log_dict, scores_dict  = self.modified_train_one_iter(batch)

                clip_frac.append(log_dict['clip_frac'])
                kl.append(log_dict['approx_kl'])
                rewards.append(log_dict['reward'])

                meteor_scores.extend(scores_dict['meteor_scores'])
                emotion_scores.extend(scores_dict['emotion_scores'])
                jacc_scores.extend(scores_dict['jacc_scores'])
                persuasion_scores.extend(scores_dict['persuasion_scores'])

                emotion_actual_probs.extend(scores_dict['emotion_actual_prob'])
                emotion_other_probs.extend(scores_dict['emotion_other_prob'])
                
                per_actual_probs.extend(scores_dict['persuasion_actual_prob'])
                per_other_probs.extend(scores_dict['persuasion_other_prob'])
                

                np.save(self.statsfolder + '/' + 'meteor_scores.npy', np.array(meteor_scores))
                np.save(self.statsfolder + '/' + 'jacc_scores.npy', np.array(jacc_scores))
                if self.classifier:
                    np.save(self.statsfolder + '/' + 'emotion_scores.npy', np.array(emotion_scores))
                    np.save(self.statsfolder + '/' + 'emotion_actual_prob.npy', np.array(emotion_actual_probs))
                    np.save(self.statsfolder + '/' + 'emotion_other_prob.npy', np.array(emotion_other_probs))
                if self.persuasion_classifier:
                    np.save(self.statsfolder + '/' + 'persuasion_scores.npy', np.array(persuasion_scores))
                    np.save(self.statsfolder + '/' + 'persuasion_actual_prob.npy', np.array(per_actual_probs))
                    np.save(self.statsfolder + '/' + 'persuasion_other_prob.npy', np.array(per_other_probs))
                update_count += 1

                if  update_count % self.evaluate_every == 0:
                    #ppl, loss, average_length, percent_strategy = self.validate_model(self.val_dataloader)
                    ppl, loss = self.validate_model(self.val_dataloader)
                    if best_ppl is None:
                        best_ppl = ppl
                        iters = update_count
                        #strategies = percent_strategy
                        #length = average_length
                        self.save_models(iters)
                        print(f'Saving Model at {iters}')
                        
                        '''filename = self.statsfolder + '/strategy_count_num_dict.json'
                        with open(filename, 'w') as f:
                            json.dump(self.count_dict_num, f)

                        filename = self.statsfolder + '/strategy_count_str_dict.json'
                        with open(filename, 'w') as f:
                            json.dump(self.count_dict_str, f)'''
                    else:
                        if ppl < best_ppl:
                            best_ppl = ppl
                            iters = update_count
                            #strategies = percent_strategy
                            #length = average_length
                            self.save_models(iters)
                            print(f'Saving Model at {iters}')
                            
                            '''filename = self.statsfolder + '/strategy_count_num_dict.json'
                            with open(filename, 'w') as f:
                                json.dump(self.count_dict_num, f)

                            filename = self.statsfolder + '/strategy_count_str_dict.json'
                            with open(filename, 'w') as f:
                                json.dump(self.count_dict_str, f)'''
                
                    print('\n')
                    #print(f'Best Perplexity Found so far {best_ppl} with % of candidates with strategy {strategies} and Average length {length} for iteration: {iters}')
                    print(f'Best Perplexity Found so far {best_ppl} for iteration: {iters}')
                    print('\n')
                    
                    val_ppl.append(ppl)
                    val_loss.append(loss)
                    #candidate_lengths.append(average_length)
                    #percent_candidates_with_strategy.append(percent_strategy)
                                
                    np.save(self.statsfolder + '/' + 'val_PPL_iter'  + '.npy', np.array(val_ppl))
                    #np.save(self.statsfolder + '/' + 'val_cand_length'  + '.npy', np.array(average_length))
                    #np.save(self.statsfolder + '/' + 'val_percent_strategy' + '.npy', np.array(percent_candidates_with_strategy))
                    np.save(self.statsfolder + '/' + 'train_rewards' + '.npy', np.array(rewards))
                    np.save(self.statsfolder + '/' + 'train_kl' + '.npy', np.array(kl))
                    np.save(self.statsfolder + '/' + 'train_clip_frac' + '.npy', np.array(clip_frac))
                    np.save(self.statsfolder + '/' + 'best_ppl_iteration_value' + '.npy', np.array(iters))
                    #np.save(self.statsfolder + '/' + 'best_ppl_percent_strategy' + '.npy', np.array(strategies))

                    #self.initialize_strategy_count()
    
                    if not self.train_single_model:
                        self.model_A.train()
                        self.model_B.train()
                    else:
                        if self.single_model_to_train == 'persuader':
                            self.model_A.train()
                if update_count == 17:
                    return best_ppl, iters

if __name__ == '__main__':
    trainer = Trainer(#modelname='Emotion_reward_2Cand_NoPersuasion_afterGridsearch',
                      modelname='Best_Model_ALL_rewards_gridsearch',
                      csvfile='dataset/Part2_fully_annotated.csv',
                      emo_classifier_filename='../emotion_classifier/roberta_model_32/best_accuracy_model.pt', #'../../kshitij/outputs',
                      device='cuda:1',
                      n_epochs=1,
                      batch_size=1,
                      mini_batch=20,
                      train_single_model=True,
                      single_model_to_train= 'persuader',
                      num_candidates=2,
                      recompute_log_prob=True,
                      average_sent_loss=True,
                      max_candidate_length=50,
                      human_reward=10,
                      beta=2.0,
                      beta2=2.0,
                      top_p=0.9,
                      temperature=0.8,
                      use_recent_past=True,
                      warmup_steps=10,
                      print_every=1,
                      evaluate_every=1,
                      learning_rate=2e-5,
                      epsilon=0.2,
                      loadModel=True,
                      loadFilename='../ARDM/medium/ARDM_best_model.pth',
                      pad_token_id=2,
                      seedvalue=10, # 10 should be the seed value since pre trained on the same seed. 
                      use_emo_classifier=True,
                      use_persuasion_classifier=True, 
                      per_classifier_filename= '../persuasion_classifier/results/roberta/32/lowest_eval_model.pt',
                      bin_classifier_filename='../persuasion_classifier/models/roberta/32/lowest_eval_model.pt',
                      per_num_labels=11,
                      emo_num_labels=23,
                      use_bleu=False,
                      use_jaccard=True,
                      use_cosine=False,
                      use_meteor=True,
                      gamma1=0.1,
                      gamma2=0.1,
                      gamma3=0.1,
                      gamma4=0.7)
    trainer.train()
