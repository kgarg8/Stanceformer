import pandas as pd, csv, pickle, pdb
from tqdm import tqdm
from openai import OpenAI
from string import Template
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--model_id', type=str, default='gpt-3.5-turbo')
parser.add_argument('--generation_txt_path', type=str, default='output_text_chatgpt_3.5_turbo.txt')
parser.add_argument('--generation_csv_path', type=str, default='output_text_chatgpt_3.5_turbo.csv')
parser.add_argument('--pickle_file_path', type=str, default='response_text_chatgpt_3.5_turbo.pkl')
parser.add_argument('--data_file', type=str, default='data/SemEval2016/raw_test_all_onecol.csv')
parser.add_argument('--limit', type=int, default=-1)
parser.add_argument('--max_new_tokens', type=int, default=200)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--prompt', type=str, default='chat-type1')

args = parser.parse_args()
print(args)

client = OpenAI()

def ChatEngine(prompt):
    response = client.chat.completions.create(
    model=args.model_id,

    max_tokens=args.max_new_tokens,
    seed=args.seed,
    messages=[
        # {"role": "system", "content": instruction},
        {"role": "user", "content": prompt}
    ]
    )
    return response


# load data
df = pd.read_csv(args.data_file, usecols=[0,1,2], encoding='ISO-8859-1')
df.columns = ['Tweet', 'Target', 'Stance']

if args.limit != -1:
    total = args.limit
else:
    total = len(df)

# prepare prompt template
if args.prompt == 'chat-type1': # zero-shot
    prompt_template = Template("Following is a tweet. $text.\n\nPlease predict the stance of the tweet towards target '$target'. Select from 'FAVOR', 'AGAINST' or 'NONE'")
elif args.prompt == 'chat-type2': # zero-shot
    prompt_template = Template("Following is a tweet. $text.\n\nPlease predict the stance in the tweet towards the target $target. Answer in the form of pythonic dictionary. {'Stance': FAVOR/ AGAINST/NONE}.")

data_to_write = [
    {"key1": "value1", "key2": "value2"},
    {"key1": "value3", "key2": "value4"}
]

# write to txt, csv files
fileEmpty = True
with open(args.generation_csv_path, 'w') as csv_file, open(args.generation_txt_path, 'w') as txt_file:

    writer = csv.writer(csv_file)
    header = ['Tweet', 'Target', 'GT Stance', 'Prompt', 'Generated Text', 'Generated Stance']
    if fileEmpty: # write header only once
        writer.writerow(header)
        fileEmpty = False

    # run model for each sample
    tweets = []; targets = []; stances = []; prompts = []; responses = []
    for index, row in tqdm(df.iterrows()):
        prompt = prompt_template.substitute(text=row['Tweet'], target=row['Target'])
        prompts.append(prompt)
        # print(prompt)
        if args.limit > 0 and index >= args.limit:
            break
        response = ChatEngine(prompt)
        # response = data_to_write[index]
        responses.append(response)
        gen_text = response.choices[0].message.content
        txt_file.write('Generated text: ' + gen_text + '\n')
        gen_tgt = ''
        writer.writerow([row['Tweet'], row['Target'], row['Stance'], prompt, gen_text, ''])
    
# dump all responses
with open(args.pickle_file_path, 'wb') as file:
    pickle.dump(responses, file)