import pandas as pd
from nltk import sent_tokenize
import matplotlib.pyplot as plt


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

avg_q = data.query('`Is Q` == True')['Word Count'].mean()
avg_e = data.query('`Is Ex` == True')['Word Count'].mean()
avg_n = data.query('`Is Norm` == True')['Word Count'].mean()

# print(data.head())
counts = [data['Is Norm'].sum(), data['Is Ex'].sum(), data['Is Q'].sum()]
labels = ['Normal', 'Exclamations', 'Questions']


plt.subplot(2, 1, 1 )
plt.pie(counts, labels=labels, autopct="%1.1f%%")
plt.title("Sentence Types in Shakespeare Text")


plt.subplot(2, 1, 2 )
plt.title("Words in Sent Types")
plt.bar(labels, [avg_n, avg_e, avg_q])
plt.show()


# total_sent = len(data)
# norm_perc = data['Is Norm'].mean() * 100
# q_perc = data['Is Q'].mean() * 100
# ex_perc = data['Is Ex'].mean() * 100

# summary = pd.DataFrame({
#     'Total Sentences': [total_sent],
#     '% Normal': [norm_perc],
#     '% Questions': [q_perc],
#     '% Exclamations': [ex_perc],
#     'Avg Words (N)': [avg_n],
#     'Avg Words (Q)': [avg_q],
#     'Avg Words (Ex)': [avg_e],
# })

# print(summary)
