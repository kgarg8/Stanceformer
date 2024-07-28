import csv, json, pandas as pd, os, pdb

PATH = 'data/SemEval14/'
input_file = f'{PATH}/Test/Laptops_Opinion_Test.json'
temp_file = f'{PATH}/temp_test.csv'
output_file = f'{PATH}/laptops_test.csv'

# Read data from JSON file
with open(input_file, mode='r', encoding='utf-8') as file:
    data = json.load(file)

# Convert data and write to CSV
with open(temp_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Text', 'Aspect', 'Polarity'])
    
    for item in data:
        raw_text = item['raw_words']
        for aspect in item['aspects']:
            aspect_terms = ' '.join(aspect['term'])
            polarity = aspect['polarity']
            writer.writerow([raw_text, aspect_terms, polarity])

df = pd.read_csv(temp_file)
df_cleaned = df.drop_duplicates()
df_cleaned.to_csv(output_file, index=False)
os.remove(temp_file)

print(f'Data has been written to {output_file}')