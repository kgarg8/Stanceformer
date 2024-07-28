import preprocessor as p
import re
import wordninja
import csv
import pandas as pd
import os
import numpy as np
import pdb


# labels = {
#         'Atheism'   : ['theism', 'atheism', 'athiest', 'thiest'],
#         'Feminist Movement': ['feminist', 'feminism'],
#         'Hillary Clinton': ['hillary', 'clinton'],
#         'Legalization of Abortion': ['abortion', 'abort'],
#         'Joe Biden': ['joe', 'biden', 'joseph'],
#         'Donald Trump': ['donald', 'trump'],
#         'Bernie Sanders': ['bernie', 'sanders'],
#         'face masks': ['face mask', 'mask'],
#         'fauci': ['fauci', 'anthony'],
#         'stay at home orders': ['stay at home', 'stay home', 'home order', 'stayhome', 'stayathome', 'homeorder'],
#         'school closures': ['school', 'closure'],
#         }

# negate_targets = {
#         'Atheism'   : 'Theism',
#         'Feminist Movement': 'Anti-Feminist Movement',
#         'Hillary Clinton': 'Exclude Hillary Clinton',
#         'Legalization of Abortion': 'Prolife',
#         'Joe Biden': 'Exclude Joe Biden',
#         'Donald Trump': 'Exclude Donald Trump',
#         'Bernie Sanders': 'Exclude Bernie Sanders',
#         'face masks': 'no face masks',
#         'fauci': 'exclude fauci',
#         'stay at home orders': 'no stay at home orders',
#         'school closures': 'school openings',
#         }

# negate_labels = {
#     2: 0,
#     0: 2
# }

def load_data(args, filename):

    df = pd.DataFrame()
    raw_text = pd.read_csv(filename, usecols=[0], encoding='ISO-8859-1')
    raw_target = pd.read_csv(filename, usecols=[1], encoding='ISO-8859-1')
    raw_label = pd.read_csv(filename, usecols=[2], encoding='ISO-8859-1')
    tar_label = pd.Series([1]*len(raw_text), name='ID') # dummy_id

    label = pd.DataFrame.replace(
        raw_label, ['POS', 'NEU', 'NEG'], [2, 1, 0])
    df = pd.concat([raw_text, label, raw_target], axis=1)
    df.columns = ['Text', 'Polarity', 'Aspect']

    # if (args.explicit or args.implicit) and 'test' in filename:
    #     new_df = pd.DataFrame()
    #     for index, row in df.iterrows():
    #         explicit_search_list = labels[row['Aspect']]
    #         condition = any(word in row['Text'] for word in explicit_search_list)
    #         if args.explicit and condition:
    #             new_df = new_df.append(row)
    #         elif args.implicit and not condition:
    #             new_df = new_df.append(row)
    #     df = new_df.reset_index(drop=True)

    # if 'test' in filename:
    #     if args.return_orig_targets:
    #         pass
    #     else:
    #         if args.replace_target_SD and not args.castToNLI:
    #             df['Aspect'] = df['Aspect'].sample(frac=1, ignore_index=True, random_state=1433)
    #         elif args.mask_targets:
    #             df['Aspect'] = ''
    #         elif args.random_targets:
    #             df['Aspect'] = df['Aspect'].sample(frac=1, ignore_index=True)
    #         elif args.negate_targets_all: 
    #             df['Aspect'] = df['Aspect'].map(lambda x: negate_targets.get(x, x)) # replace with values from negate_targets dict if it exists, otherwise leave the value unchanged
    #             df['Polarity'] = df['Polarity'].map(lambda x: negate_labels.get(x, x))

    df2 = pd.concat([raw_text, tar_label, label], axis=1)
    df2.columns = ['Text', 'ID', 'Polarity']
    df2 = df2[df2['Polarity'] != 1]  # remove 'none' label
    return df, df2


def data_clean(strings, norm_dict, task):
    clean_data = re.findall(r"[A-Za-z#@]+|[,.!?&/\<>=$]|[0-9]+", strings)
    clean_data = [[x.lower()] for x in clean_data]
    clean_data = [j for i in clean_data for j in i]
    return clean_data


def clean_all(args, filename, norm_dict):

    df, df2 = load_data(args, filename)  # load all data as DataFrame type

    # main task
    # convert DataFrame to list ['string','string',...]
    raw_data = df['Text'].values.tolist()
    label = df['Polarity'].values.tolist()
    x_target = df['Aspect'].values.tolist()
    clean_data = [None for _ in range(len(raw_data))]

    print(f"data size in {filename}: {len(clean_data)}")

    max_len = 0
    for i in range(len(raw_data)):
        clean_data[i] = data_clean(raw_data[i], norm_dict, 'main')
        x_target[i] = data_clean(x_target[i], norm_dict, 'main')
        if len(clean_data[i]) > max_len:
            max_len = len(clean_data[i])
    avg_ls = sum([len(x) for x in clean_data])/len(clean_data)
    print("average length: ", avg_ls)
    print("maximum length: ", max_len)

    t1 = [len(x) for x in clean_data]
    if max(t1) > 120:
        print('Watch out for text overflow!')

    # uncommment to save processed data file
    # tweets = [' '.join(item) for item in clean_data]
    # targets = [' '.join(item) for item in x_target]
    # labels = ['AGAINST','NONE','FAVOR']
    # name_labels = []
    # for item in label: name_labels.append(labels[item])
    # save_df = pd.DataFrame({'Text': tweets, 'Aspect': targets, 'Polarity': name_labels})
    # save_df.to_csv(f'{filename}_processed.csv', index=False)

    # auxiliary target prediction task
    # convert DataFrame to list ['string','string',...]
    raw_data2 = df2['Text'].values.tolist()
    label2 = df2['ID'].values.tolist()
    clean_data2 = [None for _ in range(len(raw_data2))]

    # print("data size in auxiliary task: ", len(raw_data2))
    for i in range(len(raw_data2)):
        clean_data2[i] = data_clean(raw_data2[i], norm_dict, 'aux')

    return clean_data, label, x_target, clean_data2, label2
