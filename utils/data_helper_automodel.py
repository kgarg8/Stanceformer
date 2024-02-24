import torch, gensim, numpy as np, transformers, random, json, sys, pdb
import gensim.models.keyedvectors as word2vec
from torch.utils.data import TensorDataset, DataLoader, Sampler
from transformers import BertTokenizer, AutoTokenizer, BertweetTokenizer
from torchtext import data
sys.path.append('./')
from utils import preprocessing


def build_vocab(x_train,x_val,x_test,x_train_target):
    
    model1 = word2vec.KeyedVectors.load_word2vec_format('crawl-300d-2M.bin', limit = 500000,binary=True)
    text_field = data.Field(lower=True)
    text_field.build_vocab(x_train,x_val,x_test,x_train_target,['<pad>'])
    print('Vocab len: ', len(text_field.vocab))
    word_vectors = dict()
    word_index = dict()
    ind = 0
    for word in text_field.vocab.itos:
        if word in model1.key_to_index:
            word_vectors[word] = model1[word]
        elif word == '<pad>':
            word_vectors[word] = np.zeros(300, dtype=np.float32)
        else:
            word_vectors[word] = np.random.uniform(-0.25, 0.25, 300)
        word_index[word] = ind
        ind = ind+1
    # print(len(list(word_vectors)))
    
    return word_vectors, word_index


def load_dataset(args,file,plm_model):

    #Creating Normalization Dictionary
    with open("utils/noslang_data.json", "r") as f:
        data1 = json.load(f)
    data2 = {}
    with open("utils/emnlp_dict.txt","r") as f:
        lines = f.readlines()
        for line in lines:
            row = line.split('\t')
            data2[row[0]] = row[1].rstrip()
    normalization_dict = {**data1,**data2}
    
    filename1 = 'data/'+file+'/raw_train_all_onecol.csv'
    filename2 = 'data/'+file+'/raw_val_all_onecol.csv'
    filename3 = 'data/'+file+'/'+args.eval_file
    args.return_orig_targets = False
    x_train,y_train,x_train_target,x_train2,y_train2 = preprocessing.clean_all(args,filename1,normalization_dict)
    x_val,y_val,x_val_target,_,_ = preprocessing.clean_all(args,filename2,normalization_dict)
    x_test,y_test,x_test_target,_,y_test2 = preprocessing.clean_all(args,filename3,normalization_dict)
    import copy
    args2 = copy.deepcopy(args)
    args2.return_orig_targets = True
    _,_,x_test_orig_targets,_,_ = preprocessing.clean_all(args2,filename3,normalization_dict)
    
    if plm_model.startswith('Bert'):
        x_train_all = [x_train,y_train,x_train_target]
        x_val_all = [x_val,y_val,x_val_target]
        x_test_all = [x_test,y_test,x_test_target]
        return x_train_all, x_val_all, x_test_all, x_test_orig_targets
    else:
        word_vectors, word_index = build_vocab(x_train,x_val,x_test,x_train_target)
        x_train_all = [x_train,y_train,x_train_target]
        x_val_all = [x_val,y_val,x_val_target]
        x_test_all = [x_test,y_test,x_test_target]
        
        return x_train_all, x_val_all, x_test_all, x_test_orig_targets, word_vectors