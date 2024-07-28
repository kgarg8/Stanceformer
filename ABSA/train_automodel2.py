# forked on Nov 27

import torch
import torch.nn as nn
import sys
import random
import logging
import numpy as np
import gspread
import os
import copy
import shap
import torch.nn.functional as F
import pandas as pd
import csv
import glob
import pdb

from ferret import SHAPExplainer
from utils import data_helper_automodel as data_helper, model_eval2 as model_eval
from argparse import ArgumentParser
import transformers
from tqdm import tqdm
from transformers import AdamW, AutoModelForSequenceClassification, BertweetTokenizer, BertTokenizer, AutoTokenizer
from torch.utils.data import DataLoader
from ferret import Benchmark

import warnings
warnings.filterwarnings('ignore')

parser = ArgumentParser()
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--total_epochs', type=int, default=5)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--aspect', type=str, default='')
parser.add_argument('--aux_eval', action='store_true')
parser.add_argument('--mul_task', action='store_true')
parser.add_argument('--dataset', type=str, default='SemEval14', choices=[
                    'SemEval14', 'SemEval15', 'SemEval16'])
parser.add_argument('--model', type=str, default='Bertweet')
parser.add_argument('--gs_sheet', type=str, default='ABSA_py_linked')
parser.add_argument('--eval_file', type=str, default='restaurants_test.csv')
parser.add_argument('--max_len', type=int, default=128)
parser.add_argument('--per_aspect', action='store_true', help='Enable by default for all datasets')
parser.add_argument('--print_aspectwise', action='store_true')
# robustness flags
# parser.add_argument('--castToNLI', action='store_true')
parser.add_argument('--mask_aspects', action='store_true')
parser.add_argument('--random_aspects', action='store_true')
parser.add_argument('--negate_aspects_all', action='store_true')
# parser.add_argument('--stress_test', type=str, default='')
parser.add_argument('--mask_aspect_SD', action='store_true',
                    help='Enable only if castToNLI is False')
parser.add_argument('--replace_aspect_SD', action='store_true',
                    help='Enable only if castToNLI is False')
# aum
# parser.add_argument('--aum', action='store_true')
# parser.add_argument('--forgettable', action='store_true')
# parser.add_argument('--remove_aum_samples', action='store_true')
# parser.add_argument('--aum_seed', type=float, default=141)
# parser.add_argument('--aum_threshold', type=float, default=-2.0)
# parser.add_argument('--aum_percentile', type=float, default=-1)
# parser.add_argument('--count_remove_aum_samples', type=int, default=-1)
# highlighting-transformer
parser.add_argument('--highlight', action='store_true')
parser.add_argument('--highlight_factor', type=float, default=0.0)
parser.add_argument('--mask', type=str, default='')
parser.add_argument('--mask_str', type=str, default='')
parser.add_argument('--explain', action='store_true')
# parser.add_argument('--explicit', action='store_true')
# parser.add_argument('--implicit', action='store_true')
parser.add_argument('--learnable_hf', action='store_true')
parser.add_argument('--eval_only', action='store_true')

args = parser.parse_args()
print(args)

num_labels = 3
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

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
tag = args.aspect

# set logger config
logfile = os.path.join('./model_{}/'.format(args.gs_sheet), 'logs.log')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# logging to a file
file_handler = logging.FileHandler(logfile)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s:%(levelname)s:%(message)s'))
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


class ABSADataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


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
    if args.highlight:
        l, m = len(encoded_dict['attention_mask']), len(
            encoded_dict['attention_mask'][0])
        t = [[[0 for col in range(m)] for col in range(m)]
             for row in range(l)]  # initialize
        encoded_dict['highlighting_matrix'] = t

        for i in range(len(t)):
            # input: sentence sep aspect sep <pad> <pad> ....
            # get index of first sep_token; id of sep is sep_token
            b = encoded_dict['input_ids'][i].index(sep_token)
            try:
                # get index of second sep_token
                e = b + 1 + encoded_dict['input_ids'][i][b+1:].index(sep_token)
                for j in range(b+1, e):
                    for k in range(b+1, e):
                        t[i][j][k] = args.highlight_factor
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

    if args.highlight:
        if args.mask_str != '':
            outdir = "model_{}/{}/{}/{}/{}".format(
                args.gs_sheet, args.model, args.dataset, args.highlight_factor, args.mask_str)
        else:
            outdir = "model_{}/{}/{}/{}/".format(
                args.gs_sheet, args.model, args.dataset, args.highlight_factor)
    else:
        outdir = "model_{}/{}/".format(args.gs_sheet, args.model)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    x_train_all, x_val_all, _, _ = data_helper.load_dataset(
        args, args.dataset, args.model)
    train_encodings = tokenize_function(tokenizer, x_train_all)
    # x_train_all[1] contains labels
    train_dataset = ABSADataset(train_encodings, x_train_all[1])
    val_encodings = tokenize_function(tokenizer, x_val_all)
    val_dataset = ABSADataset(val_encodings, x_val_all[1])

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False)

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
                highlighting_matrix = None
                if args.highlight:
                    highlighting_matrix = batch['highlighting_matrix'].to(
                        device)

                # we don't mask anything in the main transformers code but hm is multiplied by head_mask
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels,
                                highlighting_matrix=highlighting_matrix, head_mask=args.mask)
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
                    highlighting_matrix = None
                    if args.highlight:
                        highlighting_matrix = batch['highlighting_matrix'].to(device)
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels,
                                    highlighting_matrix=highlighting_matrix, head_mask=args.mask)
                    val_labels = torch.cat((val_labels, labels))
                    preds = torch.cat((preds, outputs.logits))

                y_pred, f1_average, _, _, accuracy = model_eval.compute_f1(
                    args, preds, val_labels, aux_eval=args.aux_eval)

                if f1_average > best_val:
                    best_val = f1_average
                    model_weight = os.path.join(
                        f'model_{args.gs_sheet}', f'{args.model}_{seed}.pt')
                    torch.save(model.state_dict(), model_weight)
                    # model.save_pretrained(outdir) # Huggingface format
        print(f'Best validation accuracy: {best_val}')
        # tokenizer.save_pretrained(outdir) # Huggingface format


def explain_dataset(tweets, labels, aspects, model, tokenizer):
    tweets = tweets[:10]
    aspects = aspects[:10]
    labels = labels[:10]
    bench = Benchmark(model, tokenizer)

    if args.model == 'Bertweet':
        sep = '</s>'
    else:
        sep = '[SEP]'

    texts = []
    preds = []
    for i in range(len(tweets)):
        text = ' '.join(tweets[i]) + f' {sep} ' + ' '.join(aspects[i])
        texts.append(text)
        pred = bench.score(text, return_dict=False).argmax(-1).tolist()
        preds.append(pred)

    name_explainers = [e.NAME for e in bench.explainers]
    pbar = tqdm(total=len(aspects), desc="explain", leave=False)
    evaluation_scores_by_explainer = {}
    for explainer in name_explainers:
        evaluation_scores_by_explainer[explainer] = {}
        for evaluator in bench.evaluators:
            evaluation_scores_by_explainer[explainer][evaluator.SHORT_NAME] = [
            ]

    for text, pred in zip(texts, preds):  # labels):

        # We generate explanations - list of explanations (one for each explainers)
        explanations = bench.explain(text, pred, show_progress=False)

        for explanation in explanations:
            # We evaluate the explanation and we obtain an ExplanationEvaluation
            evaluation = bench.evaluate_explanation(
                explanation, pred, show_progress=False)

            # We accumulate the results for each explainer
            for evaluation_score in evaluation.evaluation_scores:
                evaluation_scores_by_explainer[explanation.explainer][evaluation_score.name].append(
                    evaluation_score.score)

        pbar.update(1)

    # We compute mean and std, separately for each explainer and evaluator
    for explainer in evaluation_scores_by_explainer:
        for score_name in list(evaluation_scores_by_explainer[explainer]):
            list_scores = evaluation_scores_by_explainer[explainer][score_name]
            if list_scores:
                # Compute mean and standard deviation
                evaluation_scores_by_explainer[explainer][score_name] = (
                    np.mean(list_scores), np.std(list_scores),)
            else:
                evaluation_scores_by_explainer[explainer].pop(score_name, None)

    pbar.close()

    # We only vizualize the average
    table = pd.DataFrame(
        {
            explainer: {
                evaluator: mean_std[0] for evaluator, mean_std in inner.items()
            }
            for explainer, inner in evaluation_scores_by_explainer.items()
        }
    ).T

    # Avoid visualizing a columns with all nan (default value if plausibility could not computed)
    table = table.dropna(axis=1, how="all")

    if args.highlight:
        if args.mask_str != '': #### NEEDS A FIX FOR OUTDIR
            outdir = "model_{}/{}/{}/{}/{}".format(
                args.gs_sheet, args.model, args.dataset, args.highlight_factor, args.mask_str)
        else:
            outdir = "model_{}/{}/{}/{}/".format(
                args.gs_sheet, args.model, args.dataset, args.highlight_factor)
    else:
        outdir = "model_{}/{}/{}/".format(args.gs_sheet,
                                          args.model, args.dataset)
    table.to_csv(f'{outdir}metrics_{args.seed}.csv')

    # append standard deviation to the above csv file
    table2 = pd.DataFrame(
        {
            explainer: {
                evaluator: mean_std[1] for evaluator, mean_std in inner.items()
            }
            for explainer, inner in evaluation_scores_by_explainer.items()
        }
    ).T

    # Avoid visualizing a columns with all nan (default value if plausibility could not computed)
    table2 = table2.dropna(axis=1, how="all")
    table2.to_csv(f'{outdir}/metrics_{args.seed}.csv', mode='a', header=False)


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
    _, _, x_test_all, x_test_orig_aspects = data_helper.load_dataset(
        args, args.dataset, args.model)
    test_encodings = tokenize_function(tokenizer, x_test_all)
    test_dataset = ABSADataset(test_encodings, x_test_all[1])
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False)

    f1_result = []
    acc_result = []
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
                highlighting_matrix = None
                if args.highlight:
                    highlighting_matrix = batch['highlighting_matrix'].to(
                        device)
                # we don't mask anything in the main transformers code but hm is multiplied by head_mask
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels,
                                highlighting_matrix=highlighting_matrix, head_mask=args.mask)
                test_labels = torch.cat((test_labels, labels))
                preds = torch.cat((preds, outputs.logits))

        y_pred, f1_average, _, _, accuracy = model_eval.compute_f1(
            args, preds, test_labels, aux_eval=args.aux_eval, metric=metric)
        f1_result.append(f1_average)
        acc_result.append(accuracy)

        if args.explain:
            explain_dataset(x_test_all[0], x_test_all[1],
                            x_test_all[2], model, tokenizer)

        # output predictions
        output_string = ''
        output_string += '_mask_aspects_True' if args.mask_aspects else ''
        output_string += '_random_aspects_True' if args.random_aspects else ''
        output_string += '_negate_aspects_all_True' if args.negate_aspects_all else ''
        output_file = f'outputs/{args.dataset}/mispredictions_{args.model}_seed_{args.seed}{output_string}.txt'
        output_file2 = f'outputs/{args.dataset}/all_{args.model}_seed_{args.seed}{output_string}.txt'
        output_file3 = f'outputs/{args.dataset}/mispredictions_{args.model}_seed_{args.seed}{output_string}.csv'
        output_file4 = f'outputs/{args.dataset}/all_{args.model}_seed_{args.seed}{output_string}.csv'
        if os.path.exists(output_file):
            os.remove(output_file)
        if os.path.exists(output_file2):
            os.remove(output_file2)
        if os.path.exists(output_file3):
            os.remove(output_file3)
        if os.path.exists(output_file4):
            os.remove(output_file4)
        labels = ['NEG', 'NEU', 'POS']

        def write_to_file(args, f, i, j, k, l, m):
            f.write('%s\n' % ' '.join(i))
            f.write('Aspect: %s\n' % ' '.join(m))
            f.write('Polarity: %s\n' % labels[j])
            if not args.mask_aspects:
                f.write('New Target: %s\n' % ' '.join(k))
            else:
                f.write('Target is masked\n')
            f.write('Pred Aspect: %s\n\n\n' % labels[l])

        with open(output_file, 'w') as f, open(output_file2, 'w') as f2, open(output_file3, 'w') as f3, open(output_file4, 'w') as f4:
            writer = csv.writer(f3)
            writer2 = csv.writer(f4)
            writer.writerow(['Tweet', 'Aspect', 'Polarity', 'New Target', 'Pred Aspect'])
            writer2.writerow(['Tweet', 'Aspect', 'Polarity', 'New Target', 'Pred Aspect'])
            for i, j, k, l, m in zip(x_test_all[0], x_test_all[1], x_test_all[2], y_pred, x_test_orig_aspects):
                new_tar = ' '.join(k) if not args.mask_aspects else 'Target is Masked'
                writer2.writerow([' '.join(i), ' '.join(m), labels[j], new_tar, labels[l]])
                if j != l:
                    write_to_file(args, f, i, j, k, l, m)
                    new_tar = ' '.join(k) if not args.mask_aspects else 'Target is Masked'
                    writer.writerow([' '.join(i), ' '.join(m), labels[j], new_tar, labels[l]])

                write_to_file(args, f2, i, j, k, l, m)

    best_result_f1 = np.array(f1_result)
    best_result_f1 = np.append(best_result_f1, best_result_f1.mean())  # append mean
    print(best_result_f1)
    
    best_result_acc = np.array(acc_result)
    best_result_acc = np.append(best_result_acc, best_result_acc.mean())  # append mean
    print(best_result_acc)

    t1 = [[args.dataset], [args.model], [tag], ['F1_mac'], [args.highlight_factor]] + [[item] for item in best_result_f1] + [[item] for item in best_result_acc]
    t1 = [['full']] + t1

    next_col_char = get_next_col_char()
    wks.update(next_col_char + '2', t1)     # write from 2nd row


if not args.eval_only:
    run_classifier()
evaluation(metric='macro')