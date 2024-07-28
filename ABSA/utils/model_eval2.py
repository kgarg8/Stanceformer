# Evaluation
import torch
import torch.nn.functional as F
import numpy as np
import pdb
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def compute_f1(args, preds, y, aux_eval=False, metric='micro'):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    rounded_preds = F.softmax(preds, dim=1)
    _, indices = torch.max(rounded_preds, 1)

    y_pred = np.array(indices.cpu().numpy())
    y_true = np.array(y.cpu().numpy())

    target_avgs, target_labels = None, None
    result = precision_recall_fscore_support(y_true, y_pred, average='macro')
    print('Macro: ', result[2])
    
    f1_average = result[2]
    accuracy = accuracy_score(y_true, y_pred)

    return y_pred, f1_average, result[0], result[1], accuracy
