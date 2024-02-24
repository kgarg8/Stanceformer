import preprocessor as p
import re
import wordninja
import csv
import pandas as pd
import os
import numpy as np
import pdb


labels = {
        'Atheism'   : ['theism', 'atheism', 'athiest', 'thiest'],
        'Feminist Movement': ['feminist', 'feminism'],
        'Hillary Clinton': ['hillary', 'clinton'],
        'Legalization of Abortion': ['abortion', 'abort'],
        'Joe Biden': ['joe', 'biden', 'joseph'],
        'Donald Trump': ['donald', 'trump'],
        'Bernie Sanders': ['bernie', 'sanders'],
        'face masks': ['face mask', 'mask'],
        'fauci': ['fauci', 'anthony'],
        'stay at home orders': ['stay at home', 'stay home', 'home order', 'stayhome', 'stayathome', 'homeorder'],
        'school closures': ['school', 'closure'],
        }

def load_data(args, filename):

    df = pd.DataFrame()
    raw_text = pd.read_csv(filename, usecols=[0], encoding='ISO-8859-1')
    raw_target = pd.read_csv(filename, usecols=[1], encoding='ISO-8859-1')
    raw_label = pd.read_csv(filename, usecols=[2], encoding='ISO-8859-1')
    try:
        tar_label = pd.read_csv(filename, usecols=[3], encoding='ISO-8859-1')
    except:
        # dummy ID value; we won't do multi-tasking with VAST dataset
        tar_label = pd.Series([1]*len(raw_text), name='ID')

    label = pd.DataFrame.replace(raw_label, ['FAVOR', 'NONE', 'AGAINST'], [2, 1, 0])
    df = pd.concat([raw_text, label, raw_target], axis=1)
    df.columns = ['Tweet', 'Stance', 'Target']

    df2 = pd.concat([raw_text, tar_label, label], axis=1)
    df2.columns = ['Tweet', 'ID', 'Stance']
    df2 = df2[df2['Stance'] != 1]  # remove 'none' label
    return df, df2


def data_clean(strings, norm_dict, task):

    p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.RESERVED)
    clean_data = p.clean(strings)  # using lib to clean URL,hashtags...
    clean_data = re.sub(r"#SemST", "", clean_data)
    clean_data = re.sub(r"#semst", "", clean_data)
    clean_data = re.findall(r"[A-Za-z#@]+|[,.!?&/\<>=$]|[0-9]+", clean_data)
    clean_data = [[x.lower()] for x in clean_data]

    for i in range(len(clean_data)):
        if clean_data[i][0] in norm_dict.keys():
            clean_data[i] = norm_dict[clean_data[i][0]].split()
            continue
    clean_data = [j for i in clean_data for j in i]
    return clean_data


def clean_all(args, filename, norm_dict):

    df, df2 = load_data(args, filename)  # load all data as DataFrame type

    # main task
    # convert DataFrame to list ['string','string',...]
    raw_data = df['Tweet'].values.tolist()
    label = df['Stance'].values.tolist()
    x_target = df['Target'].values.tolist()
    clean_data = [None for _ in range(len(raw_data))]

    print(f"data size in {filename}: {len(clean_data)}")

    max_len = 0
    count_g100 = 0
    for i in range(len(raw_data)):
        clean_data[i] = data_clean(raw_data[i], norm_dict, 'main')
        x_target[i] = data_clean(x_target[i], norm_dict, 'main')
        if len(clean_data[i]) > max_len:
            max_len = len(clean_data[i])
        if args.dataset == 'Covid19' and len(clean_data[i]) > 100:
            count_g100 += 1
            clean_data[i] = clean_data[i][:100]
    avg_ls = sum([len(x) for x in clean_data])/len(clean_data)
    print("average length: ", avg_ls)
    print("maximum length: ", max_len)

    t1 = [len(x) for x in clean_data]
    if args.dataset == 'Covid19':
        print(f'Greater than 100 words: {count_g100}')
        if max(t1) > 140:
            print('Watch out for text overflow!')
    elif max(t1) > 120:
        print('Watch out for text overflow!')

    # convert DataFrame to list ['string','string',...]
    raw_data2 = df2['Tweet'].values.tolist()
    label2 = df2['ID'].values.tolist()
    clean_data2 = [None for _ in range(len(raw_data2))]

    for i in range(len(raw_data2)):
        clean_data2[i] = data_clean(raw_data2[i], norm_dict, 'aux')

    return clean_data, label, x_target, clean_data2, label2
