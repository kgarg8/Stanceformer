# created for loading new dataset (midas version)
import torch, pdb, transformers, random, numpy as np, ast, pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import Dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from string import Template
from argparse import ArgumentParser
from pathlib import Path

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

parser = ArgumentParser()
parser.add_argument('--model_id', type=str, default='EleutherAI/gpt-neox-20b')
parser.add_argument('--save_path', type=str, default='outputs')
parser.add_argument('--cache_dir', type=str, default='')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--grad_acc_steps', type=int, default=4)
parser.add_argument('--num_train_epochs', type=int, default=1)
parser.add_argument('--max_steps', type=int, default=-1, help='max_steps overrides num_train_epochs if non-negative')
parser.add_argument('--save_steps', type=int, default=500)
parser.add_argument('--save_total_limit', type=int, default=5)
parser.add_argument('--limit', type=int, default=-1)
parser.add_argument('--patience', type=int, default=1)
parser.add_argument('--src_max_len', type=int, default=4096)
parser.add_argument('--data_file', type=str, default='')
parser.add_argument('--eval_data_file', type=str, default='')
parser.add_argument('--target_modules', type=str, default='')
parser.add_argument('--prompt', type=str, default='default')

args = parser.parse_args()
print(args)

set_seed(123)

if args.target_modules == '':
    args.target_modules = None
else:
    args.target_modules = ast.literal_eval(args.target_modules)

# create directory if doesn't exist
Path(args.save_path).mkdir(parents=True, exist_ok=True)

# bits and bytes config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16
)

# load base LLM model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_auth_token=True, cache_dir=args.cache_dir)
model = AutoModelForCausalLM.from_pretrained(args.model_id, quantization_config=bnb_config, device_map={'':0}, use_auth_token=True, cache_dir=args.cache_dir)

# prepare model
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

def print_trainable_parameters(model):
    '''
    Prints the number of trainable parameters in the model.
    '''
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f'trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}'
    )

# Lora config
config = LoraConfig(
    r=8, 
    lora_alpha=32,
    target_modules=args.target_modules, 
    lora_dropout=0.05, 
    bias='none', 
    task_type='CAUSAL_LM'
)

# Load the Peft model
model = get_peft_model(model, config)
print_trainable_parameters(model)

# load datasets
df = pd.read_csv(args.data_file, usecols=[0,1,2], encoding='ISO-8859-1')
df.columns = ['Tweet', 'Target', 'Stance']

eval_df = pd.read_csv(args.eval_data_file, usecols=[0,1,2], encoding='ISO-8859-1')
eval_df.columns = ['Tweet', 'Target', 'Stance']

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if args.prompt == 'chat-type1':
    prompt_template = Template("[INST]\nConsider the following tweet and target.\nTweet: $tweet\nTarget: $tgt\nPlease predict the stance in the tweet towards the target ‘$tgt’. Answer in just 1 word (FAVOR/ AGAINST/ NONE). Please do not give justification.\nStance: [/INST] $stance</s>")

if args.limit > 0:
    df = df[:args.limit]
    eval_df = eval_df[:args.limit]

df['Tweet'] = df.apply(lambda row: prompt_template.substitute(tweet=row['Tweet'], tgt=row['Target'], stance=row['Stance']), axis=1)
data = Dataset.from_pandas(df)

eval_df['Tweet'] = eval_df.apply(lambda row: prompt_template.substitute(tweet=row['Tweet'], tgt=row['Target'], stance=row['Stance']), axis=1)
eval_data = Dataset.from_pandas(eval_df)

# tokenize data
data = data.map(lambda samples: tokenizer(samples['Tweet'], truncation=True, max_length=args.src_max_len))
eval_data = eval_data.map(lambda samples: tokenizer(samples['Tweet'], truncation=True, max_length=args.src_max_len))

trainer = transformers.Trainer(
    model=model,
    train_dataset=data,
    eval_dataset=eval_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc_steps,
        warmup_steps=2,
        # save_steps=args.save_steps,
        save_strategy='epoch',
        # save_total_limit=args.save_total_limit,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir='outputs',
        optim='paged_adamw_8bit',
        report_to='wandb',
        do_eval=True,
        evaluation_strategy='epoch',
        metric_for_best_model='eval_loss',
        load_best_model_at_end=True,
    ),
    callbacks=[transformers.EarlyStoppingCallback(early_stopping_patience=args.patience)],
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()

# Save model
model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model  # Take care of distributed/parallel training
model_to_save.save_pretrained(args.save_path)