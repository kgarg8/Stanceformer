# parse generated text for GPT-3.5
import pandas as pd, re, math, numpy as np, pdb

df = pd.read_csv('../gen_outputs_csv/output_text_chatgpt_3.5_turbo_hf_Covid19_wo_finetune_exp1.csv')
df = df.astype({'Generated Stance': 'str'})

def parse_stance(gen_text):
    # Use regex to extract any characters after [/INST] and then find {'Stance' : }
    match = re.search(r'(FAVOR|AGAINST|NONE)', gen_text, re.IGNORECASE)

    stance = ''
    if match:
        stance = match.group(0)
    else:
        print("No match found but these are NONE candidates.")
        stance = 'NONE'
    return stance

for index, row in df.iterrows():
    gen_text = row['Generated Text']
    
    # Check if gen_text is NaN
    if isinstance(gen_text, float) and (math.isnan(gen_text) or np.isnan(gen_text)):
        stance = ''
    else:
        stance = parse_stance(gen_text).upper()
    df.at[index, 'Generated Stance'] = stance

df.to_csv('df_parsed_output_text_chatgpt_3.5_turbo_hf_Covid19_wo_finetune_exp1.csv', index=False)