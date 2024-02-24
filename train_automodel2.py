# forked on Dec 1

import torch
import torch.nn as nn
import sys
import random
import logging
import numpy as np
import gspread
import os
import copy
import torch.nn.functional as F
import pandas as pd
import csv
import glob
import pdb

from utils import data_helper_automodel as data_helper, model_eval2 as model_eval
from argparse import ArgumentParser
import transformers
from tqdm import tqdm
from transformers import AdamW, AutoModelForSequenceClassification, BertweetTokenizer, BertTokenizer, AutoTokenizer
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings('ignore')

parser = ArgumentParser()
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--total_epochs', type=int, default=5)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--aux_eval', action='store_true')
parser.add_argument('--dataset', type=str, default='SemEval2016', choices=['vast', 'SemEval2016', 'Covid19', 'PStance'])
parser.add_argument('--model', type=str, default='Bertweet')
parser.add_argument('--gs_sheet', type=str, default='TSE_py_linked')
parser.add_argument('--eval_file', type=str, default='raw_test_all_onecol.csv')
parser.add_argument('--max_len', type=int, default=128)
parser.add_argument('--per_target', action='store_true')
# stanceformer
parser.add_argument('--target_awareness', action='store_true')
parser.add_argument('--alpha', type=float, default=0.0)
parser.add_argument('--mask', type=str, default='')
parser.add_argument('--mask_str', type=str, default='')

args = parser.parse_args()
print(args)

num_labels = 3
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if args.mask == '':
    args.mask = None
else:
    args.mask = torch.tensor(eval(args.mask)).to(device)

transformers.logging.set_verbosity_error()

random_seeds = [1, 2, 3]

# use gspread to record results
sa = gspread.service_account()
sh = sa.open('Results2')
wks = sh.worksheet(args.gs_sheet)

# set logger config
logfile = os.path.join('./model_{}/'.format(args.gs_sheet), 'logs.log')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# logging to a file
file_handler = logging.FileHandler(logfile)
file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(message)s'))
logger.addHandler(file_handler)
# logging to console
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(stream_handler)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def next_available_col():
    # check based on number of cols in row:2
    str_list = list(filter(None, wks.row_values(2)))
    return len(str_list)+1


def get_next_col_char():
    next_col = next_available_col()
    # get first char of column
    first_char = ''
    factor = (next_col-1)//26
    if next_col > 26:
        first_char = chr(64 + factor)
    # get second char of column
    second_char = chr(64 + (next_col - factor*26))
    # append first and second char of column
    next_col_char = first_char + second_char
    return next_col_char


class StanceDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, ids):
        self.encodings = encodings
        self.labels = labels
        self.ids = ids

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        item['ids'] = torch.tensor(self.ids[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
    def update_by_ids(self, ids_to_update, new_data):
        for i, sample_id in enumerate(self.ids):
            if sample_id in ids_to_update:
                idx = ids_to_update.index(sample_id)
                self.encodings['ta_matrix'][i] = new_data[idx]


def tokenize_function(tokenizer, examples):
    inputs = copy.deepcopy(examples[0])
    if args.model == 'Bertweet':
        sep = '</s>'
        sep_token = 2
    elif args.model == 'BertLite':
        sep = '[SEP]'
        sep_token = 3
    else:
        sep = '[SEP]'
        sep_token = 102
    # modifies examples[2] as tweet + tar
    _ = [x.extend([sep] + y) for x, y in zip(inputs, examples[2])]
    inputs = [' '.join(sample) for sample in inputs]
    encoded_dict = tokenizer.batch_encode_plus(
        inputs, add_special_tokens=True, max_length=args.max_len, padding='max_length', return_attention_mask=True, truncation=True)

    extra_long_samples = 0
    if args.target_awareness:
        l, m = len(encoded_dict['attention_mask']), len(
            encoded_dict['attention_mask'][0])
        t = [[[0 for col in range(m)] for col in range(m)]
             for row in range(l)]  # initialize
        encoded_dict['ta_matrix'] = t

        for i in range(len(t)):
            # input: sentence sep target sep <pad> <pad> ....
            # get index of first sep_token; id of sep is sep_token
            b = encoded_dict['input_ids'][i].index(sep_token)
            try:
                # get index of second sep_token
                e = b + 1 + encoded_dict['input_ids'][i][b+1:].index(sep_token)
                for j in range(b+1, e):
                    for k in range(b+1, e):
                        t[i][j][k] = args.alpha
            except:
                extra_long_samples += 1
    print('Extra long samples: ', extra_long_samples)
    return encoded_dict


def run_classifier():
    set_seed(123)  # initial seed

    logger.info("************************* Training *************************")
    logger.info("*************************" +
                args.model+"*************************")
    if args.model == 'Bertweet':
        tokenizer = BertweetTokenizer.from_pretrained(
            "vinai/bertweet-base", normalization=True, is_fast=True)
    elif args.model == 'Bert':
        tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", normalization=True, is_fast=True)
    elif args.model == 'BertCT':
        tokenizer = BertTokenizer.from_pretrained(
            "digitalepidemiologylab/covid-twitter-bert-v2", normalization=True, is_fast=True)
    elif args.model == 'BertLite':
        tokenizer = AutoTokenizer.from_pretrained("albert-base-v2", normalization=True, is_fast=True)

    if args.target_awareness:
        if args.mask_str != '':
            outdir = "model_{}/{}/{}/{}/{}".format(
                args.gs_sheet, args.model, args.dataset, args.alpha, args.mask_str)
        else:
            outdir = "model_{}/{}/{}/{}/".format(
                args.gs_sheet, args.model, args.dataset, args.alpha)
    else:
        outdir = "model_{}/{}/".format(args.gs_sheet, args.model)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    x_train_all, x_val_all, _, _ = data_helper.load_dataset(args, args.dataset, args.model)
    train_encodings = tokenize_function(tokenizer, x_train_all)
    # x_train_all[1] contains labels
    train_ids = list(range(len(train_encodings['input_ids'])))
    train_dataset = StanceDataset(train_encodings, x_train_all[1], train_ids)
    val_encodings = tokenize_function(tokenizer, x_val_all)
    val_dataset = StanceDataset(val_encodings, x_val_all[1], list(range(len(val_encodings['input_ids']))))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    for seed in random_seeds:
        args.seed = seed
        print("current random seed: ", seed)
        logger.info("current random seed: %d", seed)
        set_seed(seed)

        if args.model == 'Bertweet':
            model = AutoModelForSequenceClassification.from_pretrained(
                "vinai/bertweet-base", num_labels=num_labels, ignore_mismatched_sizes=True)
        elif args.model == 'Bert':
            model = AutoModelForSequenceClassification.from_pretrained(
                "bert-base-uncased", num_labels=num_labels, ignore_mismatched_sizes=True)
        elif args.model == 'BertCT':
            model = AutoModelForSequenceClassification.from_pretrained(
                "digitalepidemiologylab/covid-twitter-bert-v2", num_labels=num_labels, ignore_mismatched_sizes=True)
        elif args.model == 'BertLite':
            model = AutoModelForSequenceClassification.from_pretrained(
                "albert-base-v2", num_labels=num_labels, ignore_mismatched_sizes=True)

        model = model.to(device)
        optimizer = AdamW(model.parameters(), lr=args.lr)

        # training
        best_val = 0
        for epoch in tqdm(range(args.total_epochs)):
            model.train()
            for items in train_loader:
                batch = items
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                ta_matrix = None
                if args.target_awareness:
                    ta_matrix = batch['ta_matrix'].to(device)

                # we don't mask anything in the main transformers code but hm is multiplied by head_mask
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels,
                                ta_matrix=ta_matrix, head_mask=args.mask)
                loss = outputs[0]  # CrossEntropyLoss() used
                loss.backward()
                optimizer.step()

            # validation
            model.eval()
            with torch.no_grad():
                preds = torch.tensor(()).to(device)
                val_labels = torch.tensor(()).to(device)
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    ta_matrix = None
                    if args.target_awareness:
                        ta_matrix = batch['ta_matrix'].to(device)
                    # we don't mask anything in the main transformers code but hm is multiplied by head_mask
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels,
                                ta_matrix=ta_matrix, head_mask=args.mask)
                    val_labels = torch.cat((val_labels, labels))
                    preds = torch.cat((preds, outputs.logits))

                y_pred, f1_average, _, _, _, _ = model_eval.compute_f1(
                    args, preds, val_labels, aux_eval=args.aux_eval)

                if f1_average > best_val:
                    best_val = f1_average
                    model_weight = os.path.join(
                        f'model_{args.gs_sheet}', f'{args.model}_{seed}.pt')
                    torch.save(model.state_dict(), model_weight)
                    # model.save_pretrained(outdir) # Huggingface format
        print(f'Best validation accuracy: {best_val}')
        # tokenizer.save_pretrained(outdir) # Huggingface format


def evaluation(metric='micro'):
    logger.info("************************* Testing *************************")
    logger.info("*************************" +
                args.model + "*************************")
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    if args.model == 'Bertweet':
        model = AutoModelForSequenceClassification.from_pretrained(
            "vinai/bertweet-base", num_labels=num_labels, ignore_mismatched_sizes=True)
        tokenizer = BertweetTokenizer.from_pretrained(
            "vinai/bertweet-base", normalization=True, is_fast=True)
    elif args.model == 'Bert':
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=num_labels, ignore_mismatched_sizes=True)
        tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", normalization=True, is_fast=True)
    elif args.model == 'BertCT':
        model = AutoModelForSequenceClassification.from_pretrained(
            "digitalepidemiologylab/covid-twitter-bert-v2", num_labels=num_labels, ignore_mismatched_sizes=True)
        tokenizer = BertTokenizer.from_pretrained(
            "digitalepidemiologylab/covid-twitter-bert-v2", normalization=True, is_fast=True)
    elif args.model == 'BertLite':
        model = AutoModelForSequenceClassification.from_pretrained("albert-base-v2", num_labels=num_labels, ignore_mismatched_sizes=True)
        tokenizer = AutoTokenizer.from_pretrained("albert-base-v2", normalization=True, is_fast=True)

    # data loading
    _, _, x_test_all, x_test_orig_targets = data_helper.load_dataset(
        args, args.dataset, args.model)
    test_encodings = tokenize_function(tokenizer, x_test_all)
    test_dataset = StanceDataset(test_encodings, x_test_all[1], list(range(len(test_encodings['input_ids']))))
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False)

    target_avgs_mul_seeds = []
    best_micro_result = []
    for seed in random_seeds:
        args.seed = seed
        print("current random seed: ", seed)
        logger.info("current random seed: %d", seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        model.load_state_dict(torch.load(
            f'model_{args.gs_sheet}/{args.model}_{seed}.pt'))
        model = model.to(device)

        # testing
        model.eval()
        with torch.no_grad():
            preds = torch.tensor(()).to(device)
            test_labels = torch.tensor(()).to(device)
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                ta_matrix = None
                if args.target_awareness:
                    ta_matrix = batch['ta_matrix'].to(device)
                # we don't mask anything in the main transformers code but hm is multiplied by head_mask
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels,
                                ta_matrix=ta_matrix, head_mask=args.mask)
                test_labels = torch.cat((test_labels, labels))
                preds = torch.cat((preds, outputs.logits))

        y_pred, f1_average, _, _, target_f1_avgs, target_labels = model_eval.compute_f1(
            args, preds, test_labels, aux_eval=args.aux_eval, metric=metric)
        best_micro_result.append(f1_average)
        target_avgs_mul_seeds.append(target_f1_avgs)

    best_result_t = np.array(best_micro_result)
    best_result_t = np.append(
        best_result_t, best_result_t.mean())  # append mean
    print(best_result_t)

    t1 = [[args.dataset], [args.model], [tag], ['F1_mac'], [args.alpha]] + [[item]
                                                              for item in best_result_t]

    next_col_char = get_next_col_char()
    wks.update(next_col_char + '2', t1)     # write from 2nd row

    if args.per_target:
        x1 = np.array(target_avgs_mul_seeds)
        # adds another row with mean values for each target
        x2 = np.vstack([x1, x1.mean(axis=0)])
        for i in range(len(x2[0])):
            t1 = [[args.dataset], [args.model], [target_labels[i]],
                  ['F1_mac'], [args.alpha]] + [[item] for item in x2[:, i]]

            next_col_char = get_next_col_char()
            wks.update(next_col_char + '2', t1)     # write from 2nd row


run_classifier()
evaluation()
if args.dataset == 'vast':
    evaluation(metric='macro')