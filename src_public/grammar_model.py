import numpy as np
import torch
import torch.nn as nn
from pytorch_transformers import RobertaTokenizer, RobertaModel, BertTokenizer, BertModel

from util import use_cuda, from_numpy

class SentenceBert(nn.Module):
    def __init__(self, model):
        super().__init__()
        if 'roberta' in model:
            print("Roberta model: {}".format(model))
            self.tokenizer = RobertaTokenizer.from_pretrained(model)
            self.bert = RobertaModel.from_pretrained(model)
        else:
            print("Bert model: {}".format(model))
            self.tokenizer = BertTokenizer.from_pretrained(model)
            self.bert = BertModel.from_pretrained(model)
        self.dim = self.bert.pooler.dense.in_features
        self.max_len = self.bert.embeddings.position_embeddings.num_embeddings
        
        if use_cuda:
            self.cuda()
    
    def forward(self, sentences, subbatch_size = 64):
        ann_full = None
        for i in range(0, len(sentences), subbatch_size):
            ann = self.annotate(sentences[i:i+subbatch_size])
            if ann_full is None:
                ann_full = ann
            else:
                ann_full = torch.cat((ann_full, ann), dim = 0)
        return ann_full
    
    def annotate(self, sentences):
        """
        Input: list of sentences, which are strings
        Output: tensor (len(sentences), bert_dim) with sentence representations
        """
        all_input_ids = np.zeros((len(sentences), self.max_len), dtype = int)
        all_input_mask = np.zeros((len(sentences), self.max_len), dtype = int)
        max_sent = 0
        for s_num, sent in enumerate(sentences):
            input_ids = self.tokenizer.encode(sent, add_special_tokens = True)
            
            all_input_ids[s_num, :len(input_ids)] = input_ids
            all_input_mask[s_num, :len(input_ids)] = 1
            max_sent = max(max_sent, len(input_ids))
        all_input_ids = all_input_ids[:, :max_sent]
        all_input_ids = from_numpy(np.ascontiguousarray(all_input_ids))
        all_input_mask = all_input_mask[:, :max_sent]
        all_input_mask = from_numpy(np.ascontiguousarray(all_input_mask))
        
        features, _ = self.bert(all_input_ids, attention_mask = all_input_mask)
        return features[:,0] # pick out [CLS]

def weight_reset(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()

class BinaryClassifier(nn.Module):
    def __init__(self, bert_model, reinit = False):
        super().__init__()
        self.bert = SentenceBert(bert_model)
        if reinit:
            print("Reinitializing parameters!")
            self.bert.apply(weight_reset)
        self.dropout = nn.Dropout(p = 0.1)
        self.span_tip = nn.Linear(self.bert.dim, 2)
        
        self.subbatch_size = 64
        if use_cuda:
            self.cuda()
    
    def forward(self, sentences):
        return self.span_tip(self.dropout(self.bert(sentences, self.subbatch_size)))