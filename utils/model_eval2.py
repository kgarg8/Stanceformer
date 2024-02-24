# Evaluation
import torch
import torch.nn.functional as F
import numpy as np
import pdb
from sklearn.metrics import precision_recall_fscore_support


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

def compute_f1(args, preds, y, aux_eval=False, metric='micro'):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    rounded_preds = F.softmax(preds, dim=1)
    _, indices = torch.max(rounded_preds, 1)

    y_pred = np.array(indices.cpu().numpy())
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
            result = precision_recall_fscore_support(y_true, y_pred, average='micro')
            print('Micro: ', result[2])
        else:
            result = precision_recall_fscore_support(y_true, y_pred, average='macro')
            print('Macro: ', result[2])
        if args.per_target: # although flag is per_target, we do per_class for VAST
            target_avgs = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0,1,2])[2]
            # t1_avg = (target_avgs[0] + target_avgs[1] + target_avgs[2]) / 3 # same as result[2]
            target_labels = ['Against', 'None', 'Favor']
                
        f1_average = result[2]

    if args.per_target and args.dataset != 'vast':
        target_avgs, target_labels = eval_per_target(args, y_true, y_pred)
    return y_pred, f1_average, result[0], result[1], target_avgs, target_labels
