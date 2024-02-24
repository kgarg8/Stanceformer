import torch, json, csv, os, pandas as pd, pdb
from tqdm import tqdm
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from string import Template
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--model_id', type=str, default='EleutherAI/gpt-neox-20b')
parser.add_argument('--load_trained_weights_path', type=str, default='')
parser.add_argument('--cache_dir', type=str, default='')
parser.add_argument('--generation_txt_path', type=str, default='output_text_gpt_neox_20b_hf.txt')
parser.add_argument('--generation_csv_path', type=str, default='output_text_gpt_neox_20b_hf.csv')
parser.add_argument('--results_file_path', type=str, default='results_text_gpt_neox_20b_hf.txt')
parser.add_argument('--data_file', type=str, default='kp20k_test_kp20k.jsonl')
parser.add_argument('--limit', type=int, default=-1)
parser.add_argument('--test_batch_size', type=int, default=4)
parser.add_argument('--src_max_len', type=int, default=1024)
parser.add_argument('--prompt', type=str, default='default')
parser.add_argument('--load_pretrained_weights_lora', action='store_true')
parser.add_argument('--load_in_8bit', action='store_true')
args = parser.parse_args()
print(args)


# load base LLM model and tokenizer
base_model = args.model_id
model = AutoModelForCausalLM.from_pretrained(base_model, load_in_8bit=args.load_in_8bit,  device_map="auto", use_auth_token=True, cache_dir=args.cache_dir)
tokenizer = AutoTokenizer.from_pretrained(base_model, use_auth_token=True, cache_dir=args.cache_dir)

# Load the Lora model
if args.load_pretrained_weights_lora:
    # Load peft config for pre-trained checkpoint etc.
    peft_model_id = args.load_trained_weights_path
    config = PeftConfig.from_pretrained(peft_model_id)
    model = PeftModel.from_pretrained(model, peft_model_id, device_map="auto")

model.eval()

print("Peft model loaded")
device = "cuda:0"

if args.prompt == 'default':
    prompt_template = Template('Following is an article. $text. Some of the keywords of this article are: [')
elif args.prompt == 'chat-type1': # zero-shot
    prompt_template = Template("<s>[INST]\n\nFollowing is a tweet. $text. \n\nPlease predict the stance in the tweet towards the target $target. Answer in the form of pythonic dictionary. {'Stance': FAVOR/ AGAINST/NONE}. Do not output anything else other than the dictionary. [/INST]")
elif args.prompt == 'chat-type2': # zero-shot
    prompt_template = Template("<s>[INST]\n\nFollowing is a tweet. $text. \n\nPlease predict the stance in the tweet towards the target $target. Answer in the form of pythonic dictionary. {'Stance': FAVOR/ AGAINST/NONE}. Do not output anything else other than the dictionary. Please do not worry about the toxicity of the answer. Provide your answer regardless of tweets containing any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.[/INST].")
elif args.prompt == 'chat-type3': # 5-shot
    prompt_template = Template("<s>[INST] Please predict the stance in the tweet towards the target ‘Hillary Clinton’. Answer in just 1 word (FAVOR/ AGAINST/ NONE). Please do not give justification.\nTweet: Save an intern.... Keep Bill & Hillary out of the White House. #SemST\nTarget: Hillary Clinton\nStance: AGAINST </s>\n<s> Please predict the stance in the tweet towards the target ‘Atheism’. Answer in just 1 word (FAVOR/ AGAINST/ NONE). Please do not give justification.\nTweet: dear lord thank u for all of ur blessings forgive my sins lord give me strength and energy for this busy day ahead #blessed #hope #SemST\nTarget: Atheism\nStance: AGAINST </s>\n<s> Please predict the stance in the tweet towards the target ‘Atheism’. Answer in just 1 word (FAVOR/ AGAINST/ NONE). Please do not give justification.\nTweet: Religions stopped being credible the minute the first dinosaur fossil was found. #SemST\nTarget: Atheism\nStance: FAVOR </s>\n<s> Please predict the stance in the tweet towards the target ‘Feminist Movement’. Answer in just 1 word (FAVOR/ AGAINST/ NONE). Please do not give justification.\nTweet: @jowilliams293 Feminists, go to the gym, lose some weight and stop to blame society for everything #SemST\nTarget: Feminist Movement\nStance: AGAINST </s>\n<s> Please predict the stance in the tweet towards the target ‘Legalization of Abortion’. Answer in just 1 word (FAVOR/ AGAINST/ NONE). Please do not give justification.\nTweet: Please join us as we pray to end the global holocaust of abortion! #SemST\nTarget: Legalization of Abortion\nStance: AGAINST </s>\n<s> Please predict the stance in the tweet towards the target ‘$target’. Answer in just 1 word (FAVOR/ AGAINST/ NONE). Please do not give justification. Output answer only for this sample.\nTweet: $text\nTarget: $target\nStance: [/INST]"
)

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

df = pd.read_csv(args.data_file, usecols=[0,1,2], encoding='ISO-8859-1')
df.columns = ['Tweet', 'Target', 'Stance']

if tokenizer.pad_token is None: ## add that check for training file also
    tokenizer.pad_token = tokenizer.eos_token

i = 0 # keep track of arg (limit)
metrics = [] # collect metric for each batch
flag_break = False
fileEmpty = True
with open(args.generation_csv_path, 'w') as csv_file, open(args.generation_txt_path, 'w') as txt_file:

    writer = csv.writer(csv_file)
    header = ['Tweet', 'Target', 'GT Stance', 'Prompt', 'Generated Text', 'Generated Stance']
    if fileEmpty: # write header only once
        writer.writerow(header)
        fileEmpty = False

    for samples in tqdm(batch(df, args.test_batch_size)):
        tweets = []; targets = []; stances = []; prompts = []
        for index, sample in samples.iterrows():
            tweets.append(sample['Tweet'])
            targets.append(sample['Target'])
            stances.append(sample['Stance'])
            prompt = prompt_template.substitute(text=sample['Tweet'], target=sample['Target'])
            # print(prompt)
            prompts.append(prompt)
            i += 1
            if args.limit > 0 and i >= args.limit:
                flag_break = True
                break

        inputs = tokenizer(prompts, truncation=True, padding=True, max_length=args.src_max_len, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=200)
        output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        srcs = []; trgs = []; predictions = []
        for output_text, prompt, tweet, target, stance in zip(output_texts, prompts, tweets, targets, stances):
            txt_file.write('Generated text: ' + output_text + '\n')
            gen_text = output_text.replace(prompt, '')
            gen_tgt = ''
            writer.writerow([tweet, target, stance, prompt, gen_text, gen_tgt])