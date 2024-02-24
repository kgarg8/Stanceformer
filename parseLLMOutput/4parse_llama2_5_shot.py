# parse generated text for llama-2 (5-shot)
import pandas as pd, re, pdb

dataset = 'vast'
df = pd.read_csv(f'../gen_outputs_csv/output_text_llama2_13b_chat_hf_{dataset}_wo_finetune_exp6.csv')
df = df.astype({'Generated Stance': 'str'})

def parse_stance(tweet_text):
    # Use regex to extract any characters after [/INST] and then find {'Stance' : }
    # Added since many outputs contained the string 'No Stance'
    match = re.search(r'\[\/INST\](.*?)(FAVOR|AGAINST|NONE|No\s?Stance|NEUTRAL)', tweet_text, re.DOTALL | re.IGNORECASE)

    stance = ''
    if match:
        preceding_text = match.group(1).strip()  # Any characters between [/INST] and {'Stance':
        stance_group_2 = match.group(2)

        if stance_group_2:
            stance = stance_group_2
            stance = stance.lower().replace('no stance', 'NONE')
            stance = stance.lower().replace('nostance', 'NONE')
            stance = stance.lower().replace('neutral', 'NONE')
        print(stance)
    else:
        # The model abstains from outputting the stance due to ethical considerations
        print("No match found.")
    return stance

for index, row in df.iterrows():
    gen_text = row['Generated Text']
    stance = parse_stance(gen_text)
    df.at[index, 'Generated Stance'] = stance.upper() # note the upper since lowercased also matched previously

new_df = df[df['Generated Stance'] == '']
print('NO MATCH FOUND count: ', len(new_df))

pdb.set_trace()
new_df.to_csv(f'parsedOutput/new_df_parsed_output_text_llama2_13b_chat_hf_{dataset}_wo_finetune_exp6.csv', index=False)

df.to_csv(f'parsedOutput/parsed_output_text_llama2_13b_chat_hf_{dataset}_wo_finetune_exp6.csv', index=False)