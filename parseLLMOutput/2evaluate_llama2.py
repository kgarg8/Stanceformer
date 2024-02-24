# Evaluation: llama2 (zero & few-shot)
import torch
import torch.nn.functional as F
import numpy as np
import pdb
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


class args:
    dataset = 'vast'
    per_target = True
    zero_shot = True

df = pd.read_csv(f'parsedOutput/parsed_output_text_llama2_7b_chat_hf_{args.dataset}_wo_finetune_exp1.csv')
df = df.astype({'Generated Stance': 'str'})

pdb.set_trace()

def eval_per_target(args, y_true, y_pred):

    # SE
    # atheism: 2-221 -> [0:220]
    # feminist movement: 222:506 -> [220:505]
    # hillary clinton: 507:801 -> [505:800]
    # legalization of abortion: 802:1081 -> [800:1080]

    # C19
    # face masks: 2:201 -> [0:200]
    # fauci: 202:401 -> [200:400]
    # stay at home: 402:601 -> [400:600]
    # school closures: 602:801 -> [600:800]

    # PS
    # DT: 2:778 -> [0:777]
    # joe biden: 779:1523 -> [777:1522]
    # BS: 1524:2158 -> [1522:2157]

    f1_data, target_labels = [], []
    if args.dataset == 'SemEval2016':
        t1 = precision_recall_fscore_support(y_true[:220], y_pred[:220], average=None, labels=[0, 1, 2])
        t1_avg = (t1[2][0] + t1[2][2]) / 2
        f1_data.append(t1_avg)
        t1 = precision_recall_fscore_support(y_true[220:505], y_pred[220:505], average=None, labels=[0, 1, 2])
        t1_avg = (t1[2][0] + t1[2][2]) / 2
        f1_data.append(t1_avg)
        t1 = precision_recall_fscore_support(y_true[505:800], y_pred[505:800], average=None, labels=[0, 1, 2])
        t1_avg = (t1[2][0] + t1[2][2]) / 2
        f1_data.append(t1_avg)
        t1 = precision_recall_fscore_support(y_true[800:1080], y_pred[800:1080], average=None, labels=[0, 1, 2])
        t1_avg = (t1[2][0] + t1[2][2]) / 2
        f1_data.append(t1_avg)
        target_labels = ['atheism', 'feminist', 'hillary', 'abortion']
    
    elif args.dataset == 'Covid19':
        t1 = precision_recall_fscore_support(y_true[0:200], y_pred[0:200], average=None, labels=[0, 1, 2])
        t1_avg = (t1[2][0] + t1[2][1] + t1[2][2]) / 3
        f1_data.append(t1_avg)
        t1 = precision_recall_fscore_support(y_true[200:400], y_pred[200:400], average=None, labels=[0, 1, 2])
        t1_avg = (t1[2][0] + t1[2][1] + t1[2][2]) / 3
        f1_data.append(t1_avg)
        t1 = precision_recall_fscore_support(y_true[400:600], y_pred[400:600], average=None, labels=[0, 1, 2])
        t1_avg = (t1[2][0] + t1[2][1] + t1[2][2]) / 3
        f1_data.append(t1_avg)
        t1 = precision_recall_fscore_support(y_true[600:800], y_pred[600:800], average=None, labels=[0, 1, 2])
        t1_avg = (t1[2][0] + t1[2][1] + t1[2][2]) / 3
        f1_data.append(t1_avg)
        target_labels = ['face', 'fauci', 'stay', 'school']
    
    elif args.dataset == 'PStance':
        t1 = precision_recall_fscore_support(y_true[0:777], y_pred[0:777], average=None, labels=[0, 1, 2])
        t1_avg = (t1[2][0] + t1[2][2]) / 2
        f1_data.append(t1_avg)
        t1 = precision_recall_fscore_support(y_true[777:1522], y_pred[777:1522], average=None, labels=[0, 1, 2])
        t1_avg = (t1[2][0] + t1[2][2]) / 2
        f1_data.append(t1_avg)
        t1 = precision_recall_fscore_support(y_true[1522:2157], y_pred[1522:2157], average=None, labels=[0, 1, 2])
        t1_avg = (t1[2][0] + t1[2][2]) / 2
        f1_data.append(t1_avg)
        target_labels = ['donald', 'joe', 'bernie']
    
    return f1_data, target_labels

def get_zero_shot_dataset(df):
    # get the zero-shot indices from the raw_test file
    df2 = pd.read_csv('../data/vast_raw/vast_test.csv')
    
    # need to ignore certain rows which differ by one special character (maybe due to different file encodings)
    indices_to_ignore = [1758, 1759, 1760, 1761, 2882, 2883, 2884]
    mask_df = ~df.index.isin(indices_to_ignore)
    mask_df2 = ~df2.index.isin(indices_to_ignore)

    # checks
    assert len(df) == len(df2), "DataFrames df and df2 have different number of rows."
    assert all(df.loc[mask_df, 'Tweet'] == df2.loc[mask_df2, 'post']), "Values in 'Tweet' column of df do not match values in 'post' column of df2."

    # get the zero-shot indices
    indices = df2.index[df2['seen?'] == 0].tolist()

    assert len(indices) == 1460, "All zero-shot samples not present"

    # get the df rows corresponding to the zero-shot indices
    df = df.iloc[indices]
    df = df.reset_index(drop=True)

    pdb.set_trace()
    return df


def compute_f1(args, preds, y, aux_eval=False, metric='micro'):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    # rounded_preds = F.softmax(preds, dim=1)
    # _, indices = torch.max(rounded_preds, 1)

    y_pred = np.array(preds.cpu().numpy())
    y_true = np.array(y.cpu().numpy())

    target_avgs, target_labels = None, None
    if not aux_eval:
        print('aux_eval=False')
        result = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0, 1, 2])
        if args.dataset == 'Covid19':
            f1_average = (result[2][0]+result[2][1]+result[2][2])/3
        else: # PStance, SemEval-2016
            f1_average = (result[2][0]+result[2][2])/2
    else:
        print('aux_eval=True')  # only for VAST
        if metric == 'micro':
            result = precision_recall_fscore_support(y_true, y_pred, average='micro', labels=[0,1,2])
            print('Micro: ', result[2])
        else:
            result = precision_recall_fscore_support(y_true, y_pred, average='macro', labels=[0,1,2])
            print('Macro: ', result[2])
        if args.per_target: # although flag is per_target, we do per_class for VAST
            target_avgs = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0,1,2])[2]
            # t1_avg = (target_avgs[0] + target_avgs[1] + target_avgs[2]) / 3 # same as result[2]
            target_labels = ['Against', 'None', 'Favor']
                
        f1_average = result[2]

    if args.per_target and args.dataset != 'vast':
        target_avgs, target_labels = eval_per_target(args, y_true, y_pred)
    
    if args.per_target:
        for i in range(len(target_avgs)):
            print(f'{target_labels[i]}: {target_avgs[i]}')
    
    print(f1_average)
    return y_pred, f1_average, result[0], result[1], target_avgs, target_labels


if args.dataset == 'vast' and args.zero_shot==True:
    df = get_zero_shot_dataset(df)

# Replace nans with pandas-nans
df['Generated Stance'].replace('nan', pd.NA, inplace=True)

# Replace NaN values in 'Generated Stance' column with 'UNKNOWN'
df['Generated Stance'].fillna('UNKNOWN', inplace=True)

# Convert text labels to int labels
df['Generated Stance'] = df['Generated Stance'].map({'FAVOR': 2, 'NONE': 1, 'AGAINST': 0, 'UNKNOWN': -1})

unknown = len(df[df['Generated Stance'] == -1])

df['GT Stance'] = df['GT Stance'].map({'FAVOR': 2, 'NONE': 1, 'AGAINST': 0}) # UNKNOWN is not in this dictionary

preds = torch.tensor(df['Generated Stance'])
y = torch.tensor(df['GT Stance'])

if args.dataset == 'vast':
    compute_f1(args, preds, y, aux_eval=True, metric='macro')
else:
    compute_f1(args, preds, y)