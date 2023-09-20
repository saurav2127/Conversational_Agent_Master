import torch
import os
from sklearn.metrics import f1_score, accuracy_score
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, XLNetTokenizer, XLNetModel, XLNetLMHeadModel, XLNetConfig, XLNetForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification
#from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random
from tqdm import tqdm, trange
import pdb


class Trainer():
    def __init__(self, traincsvpath, testcsvpath, valcsvpath, device, batch_sizes, num_epochs, modelnames,
                 num_labels, model_save_path, accuracy_save_path, learning_rate):
        self.seed()
        self.traincsvpath = traincsvpath
        self.valcsvpath = valcsvpath
        self.testcsvpath = testcsvpath
        self.batch_sizes = batch_sizes
        self.num_epoch = num_epochs
        self.modelnames = modelnames
        self.model_save_path = model_save_path
        self.accuracy_save_path = accuracy_save_path
        self.learning_rate = learning_rate
        self.num_labels = num_labels
        #self.lowest_eval_loss = None
        #self.best_accuracy = None
        self.num_epochs = num_epochs
        self.start_epoch = 0
        self.device = device

    def initialize_dataloaders(self):
        self.train_dataloader = self.get_dataloader(self.traincsvpath, train=True)
        self.valid_dataloader = self.get_dataloader(self.valcsvpath, train=False)
        self.test_dataloader = self.get_dataloader(self.testcsvpath, train=False)

    def create_stat_dict(self):
        self.stats_dict = {}
        for modelname in self.modelnames:
            self.stats_dict[str(modelname)] = {}
            for batch_size in self.batch_sizes:
                self.stats_dict[str(modelname)][str(batch_size)] = {}
                self.stats_dict[str(modelname)][str(batch_size)]['macro'] = []
                self.stats_dict[str(modelname)][str(batch_size)] ['average'] = []

    def config(self):
        configs = {'batch_size': self.batch_size,
                   'model_name': self.modelname,
                   'learning_rate': self.learning_rate,
                   }
        path = 'config_' + self.batch_size + self.modelname + '.json'
        with open(path, 'w') as j:
            json.dump(configs, j)

    def load_dataset(self, path):
        df = pd.read_csv(path, index_col=0)
        text = df['Unit'].tolist()
        label = df['binary_label'].tolist()
        return text, label

    def clean_text(self, x):
        cleaned = ''
        x = re.sub("@\S+", " ", x) # remove @ mentions
        x = re.sub("https*\S+", " ", x) # remove URL
        x = re.sub("#\S+", " ", x) # remove hastags
        text = re.sub(r"[^a-zA-z.!?'0-9]", ' ', text)
        text = re.sub('\t', ' ',  text)
        text = re.sub(r" +", ' ', text)
        text = text.strip()
        text = text.split()
        for i in text:
            cleaned += i
            cleaned += ' '
        return cleaned

    def preprocess_text(self, text:list):
        cleaned_text = []
        for i in text:
            cleaned_text.append(clean_test(i))
        return cleaned_text

    def get_dataloader(self, path, train:bool):
        text, labels = self.get_data(path)
        dataloader = self.dataloader(text, labels, train)
        return dataloader

    def tokenize_inputs(self, text):
        input_ids = []
        attention_mask = []
        for i in text:
            self.tokenizer()

    def get_data(self, path):
        text, labels = self.load_dataset(path)
        #text = self.preproces_text(text)
        #pdb.set_trace()
        text = self.tokenizer(text, padding=True, truncation=True,  max_length=50, return_tensors='pt')
        return text, labels

    def dataloader(self, text, labels, train:bool):
        data = TensorDataset(text['input_ids'], text['attention_mask'], torch.tensor(labels))
        if train:
            sampler = RandomSampler(data)
            dataloader = DataLoader(data,
                                    sampler=sampler,
                                    batch_size=self.batch_size)
        else:
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data,
                                    sampler=sampler,
                                    batch_size=self.batch_size)
        return dataloader

    def initialize_optimizer(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
                                        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                                        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay':0.0}
                                        ]
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)

    def validate(self, epoch):
        eval_loss = 0
        num_eval_samples = 0
        pred_labels = []
        true_labels = []
        for idx, batch in enumerate(self.valid_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                output = self.model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                probs = F.softmax(output.logits, dim=1)
                _, pred_label = torch.topk(probs, k=1, dim=-1)
                pred_labels.extend(pred_label.squeeze(1).tolist())
                true_labels.extend(b_labels.tolist())
                eval_loss += output.loss.item()
                num_eval_samples += b_labels.size(0)

        epoch_eval_loss = eval_loss/num_eval_samples
        self.valid_loss_set.append(epoch_eval_loss)

        micro = f1_score(true_labels, pred_labels, average='micro')
        macro = f1_score(true_labels, pred_labels, average='macro')
        accuracy = accuracy_score(true_labels, pred_labels)

        self.micro_scores.append(micro)
        self.macro_scores.append(macro)
        self.accuracy_scores.append(accuracy)

        print("Valid loss: {}".format(epoch_eval_loss))
        print(f"Macro F1 score {macro} and  Accuracy {accuracy}")

        if self.lowest_eval_loss == None:
            self.lowest_eval_loss = epoch_eval_loss
            self.save_model(epoch, micro, macro, accuracy, self.model_save_path + '/lowest_eval_model.pt')
        else:
            if epoch_eval_loss < self.lowest_eval_loss:
                self.lowest_eval_loss = epoch_eval_loss
                self.save_model(epoch, micro, macro, accuracy, self.model_save_path + '/lowest_eval_model.pt')
        if self.best_accuracy == None :
            self.best_accuracy = accuracy
            self.save_model(epoch,  micro, macro, accuracy, self.model_save_path + '/best_accuracy_model.pt')
        else:
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.save_model(epoch, micro, macro, accuracy, self.model_save_path + '/best_accuracy_model.pt')
        print("\n")

    def train(self):
        best_config = {'modelname': None,
                       'batch_size': None,
                       'learning_rate': None,
                       'validation_loss': None,
                       'accuracy': None,
                       'F1_score': None}
        i = 0
        for model in self.modelnames:
            self.modelname = model
            for batchsize in self.batch_sizes:
                if i != 0:
                    self.seed()
                i += 1
                self.batch_size = batchsize
                self.initialise_model()
                self.initialize_optimizer()
                self.initialize_dataloaders()
                self.make_relevant_directory()
                #self.config()
                #self.create_stat_dict()
                self.macro_scores = []
                self.micro_scores = []
                self.accuracy_scores = []
                self.train_loss_set = []
                self.valid_loss_set = []
                self.model = self.model.to(self.device)
                self.lowest_eval_loss = None
                self.best_accuracy = None
                for i in trange(self.num_epochs, desc="Epoch"):
                    actual_epoch = self.start_epoch + i
                    self.model.train()
                    tr_loss = 0
                    num_train_samples = 0
                    for step, batch in enumerate(self.train_dataloader):
                        batch = tuple(t.to(device) for t in batch)
                        b_input_ids, b_input_mask, b_labels = batch
                        self.optimizer.zero_grad()
                        output = self.model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                        tr_loss += output.loss.item()
                        num_train_samples += b_labels.size(0)
                        output.loss.backward()
                        self.optimizer.step()
                    epoch_train_loss = tr_loss / num_train_samples
                    self.train_loss_set.append(epoch_train_loss)
                    self.model.eval()
                    self.validate(i)
                    print("Train loss after Epoch {} : {}".format(actual_epoch, epoch_train_loss))
                self.save_stats()
                print("\n")
                accuracy, macro = self.test_model(self.model_save_path + '/lowest_eval_model.pt')
                print(f"LOSS Based: Macro F1: {macro} and Accuracy: {accuracy}")
                print('\n')
                accuracy, macro = self.test_model(self.model_save_path + '/best_accuracy_model.pt')
                print(f"ACCURACY Based: Macro F1: {macro} and Accuracy: {accuracy}")
                print('\n')

    def load_model(self, save_path):
        if self.modelname =='roberta':
            self.model = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=self.num_labels)
            self.model.config.problem_type = 'single_label_classification'
        elif self.modelname == 'xlnet':
            self.model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=self.num_labels)
            self.model.config.problem_type = 'single_label_classification'
        checkpoint = torch.load(save_path)
        model_state_dict = checkpoint['state_dict']
        self.model.load_state_dict(model_state_dict)

    def save_stats(self):
        np.save(self.stats_path + '/macro.npy', self.macro_scores)
        np.save(self.stats_path + '/accuracy.npy', self.accuracy_scores)

    @torch.no_grad()
    def test_model(self, save_path):
        pred_labels = []
        true_labels = []
        self.load_model(save_path)
        self.model = self.model.to(self.device)
        for idx, batch in enumerate(self.test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                output = self.model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                probs = F.softmax(output.logits, dim=1)
                _, pred_label = torch.topk(probs, k=1, dim=-1)
                pred_labels.extend(pred_label.squeeze(1).tolist())
                true_labels.extend(b_labels.tolist())
        macro = f1_score(true_labels, pred_labels, average='macro')
        accuracy = accuracy_score(true_labels, pred_labels)
        print('\n')
        print('\n')
        print(f"TEST FOR {self.modelname} and Batch Size {self.batch_size}")
        #print(f"Accuracy: {accuracy}, Macro F1 Score: {macro}")
        return accuracy, macro

    def save_model(self, epoch,  micro, macro, accuracy, model_save_path):
        #model_to_save = model.module if hasattr(model, 'module') else model
        checkpoint = {'epoch': epoch, \
                      'lowest_eval_loss': self.lowest_eval_loss,\
                      'state_dict': self.model.state_dict(),\
                      'train_loss_hist': self.train_loss_set,\
                      'valid_loss_hist': self.valid_loss_set,\
                      'F1_micro': micro,\
                      'F1_macro': macro, \
                      'Accuracy': accuracy
                     }
        torch.save(checkpoint, model_save_path)
        print("Saving model at iteration {} with validation loss of {}".format(epoch,\
                                                                         self.lowest_eval_loss))

    def seed(self, seed=10):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    def initialise_model(self):
        if self.modelname =='roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
            self.model = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=self.num_labels)
            self.model.config.problem_type = 'single_label_classification'
        elif self.modelname == 'xlnet':
            self.model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=self.num_labels)
            self.model.config.problem_type = 'single_label_classification'
            self.tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
        else:
          self.model = AutoModelForSequenceClassification.from_pretrained('', num_labels=self.num_labels)

    def make_relevant_directory(self):
        self.model_save_path = os.path.join(os.getcwd(), 'models', self.modelname, str(self.batch_size))
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
            print(f"Model directory for {self.modelname} and batch size {self.batch_size} created!")
        else:
            print(f"Model directory for {self.modelname} and batch size {self.batch_size} already exists!")
        ## creating stats folder:
        self.stats_path = os.path.join(os.getcwd(), 'stats', self.modelname, str(self.batch_size))
        if not os.path.exists(self.stats_path):
            os.makedirs(self.stats_path)
            print(f"Stats directory for {self.modelname} and batch size {self.batch_size} created!")
        else:
            print(f"Stats directory for {self.modelname} and batch size {self.batch_size} already exists!")


if __name__ == "__main__":

    # Loading the data:
    traincsvpath = 'dataset/train_binary.csv'
    valcsvpath = 'dataset/val_binary.csv'
    testcsvpath = 'dataset/test_binary.csv'

    model_save_path = 'results/'
    accuracy_save_path = 'results/'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_sizes = [8, 16, 32, 64]
    num_epochs = 10
    learning_rate = 2e-5
    modelnames = ['roberta']

    num_labels = 2

    trainer = Trainer(traincsvpath, testcsvpath, valcsvpath, device, batch_sizes, num_epochs, modelnames, num_labels,
          model_save_path, accuracy_save_path, learning_rate)

    trainer.train()
