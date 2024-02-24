# parse generated text for llama-2 (zero-shot)
import pandas as pd, re, pdb

dataset = 'vast'
df = pd.read_csv(f'../gen_outputs_csv/output_text_llama2_7b_chat_hf_{dataset}_wo_finetune_exp1.csv')
df = df.astype({'Generated Stance': 'str'})

def parse_stance(tweet_text):
    # Use regex to extract any characters after [/INST] and then find {'Stance' : }
    match = re.search(r'\[\/INST\](.*?)(?=[\"]?Stance[\'"]?: [\'"]?(FAVOR|AGAINST|NONE)[\'"]?)', tweet_text, re.DOTALL)

    stance = ''
    if match:
        preceding_text = match.group(1).strip()  # Any characters between [/INST] and {'Stance':
        stance_group_2 = match.group(2)

        if stance_group_2:
            stance = stance_group_2
        print(stance)
    else:
        # The model abstains from outputting the stance due to ethical considerations
        print("No match found.")
    return stance

for index, row in df.iterrows():
    gen_text = row['Generated Text']
    stance = parse_stance(gen_text)
    # row['Generated Stance'] = stance
    df.at[index, 'Generated Stance'] = stance

new_df = df[df['Generated Stance'] == '']
print('NO MATCH FOUND count: ', len(new_df))

new_df.to_csv(f'parsedOutput/new_df_parsed_output_text_llama2_7b_chat_hf_{dataset}_wo_finetune_exp1.csv', index=False)

df.to_csv(f'parsedOutput/parsed_output_text_llama2_7b_chat_hf_{dataset}_wo_finetune_exp1.csv', index=False)