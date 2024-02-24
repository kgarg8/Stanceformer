import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModel, BertModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class stance_classifier(nn.Module):

    def __init__(self,num_labels,plm_model):
        super(stance_classifier, self).__init__()
        
        self.dropout = nn.Dropout(0.)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
        if plm_model == 'Bertweet':
            self.bert = AutoModel.from_pretrained("vinai/bertweet-base")
            # self.bert = AutoModel.from_pretrained("digitalepidemiologylab/covid-twitter-bert")
        elif plm_model == 'Bert':
            self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.bert.pooler = None
        self.linear = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.out = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.linear2 = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.out2 = nn.Linear(self.bert.config.hidden_size, 18)
        
    def forward(self, x_input_ids, x_seg_ids, x_atten_masks, x_len, task_id, aux_eval=False):

        last_hidden = self.bert(input_ids=x_input_ids, attention_mask=x_atten_masks, token_type_ids=x_seg_ids)
        cls = last_hidden[0][:,0]
        query = self.dropout(cls)
        
        if aux_eval:
            task_id[0] = 1
        
        if task_id[0] == 0:
            linear = self.relu(self.linear(query))
            out = self.out(linear)
        else:
            linear = self.relu(self.linear2(query))
            out = self.out2(linear)
        
        return out
