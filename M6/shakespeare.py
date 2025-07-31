import pandas as pd
from nltk import sent_tokenize


file_path = "M6/shakespeare.txt"
with open(file_path, encoding="utf-8") as f:
    play_text = f.read()

sentences = sent_tokenize(play_text)
print(type(sentences))

data = pd.DataFrame({
    'Sentence': sentences,
    'Sent Len': [len(s) for s in sentences],
    'Punct Count': [sum(1 for ch in s if ch in ".,!?;:-") 
                    for s in sentences],
    'Word Count': [len(s.split()) for s in sentences]
})

# print(data.head())

data['Is Q'] = data['Sentence'].str.endswith('?')
data['Is Ex'] = data['Sentence'].str.endswith('!')
data['Is Norm'] = ~(data['Is Q'] | data['Is Ex'])

# print(data.head())

avg_q = data.query('`Is Q` == True')['Word Count'].mean()
avg_e = data.query('`Is Ex` == True')['Word Count'].mean()
avg_n = data.query('`Is Norm` == True')['Word Count'].mean()

summary = pd.DataFrame({
    'Total Sentences': [len(data)],
    '% Normal': [data['Is Norm'].mean() * 100],
    '% Questions': [data['Is Q'].mean() * 100],
    '% Exclamations': [data['Is Ex'].mean() * 100],
    'Avg Words (N)': [avg_n],
    'Avg Words (Q)': [avg_q],
    'Avg Words (Ex)': [avg_e],
})

print(summary)
