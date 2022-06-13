import pandas as pd
from preprocess import prepr

messages, labels = prepr(pd.read_csv('data.csv'), True, stop_words=['а', 'и', 'я', 'не', 'в', 'с', 'у', 'на', 'что', 'здравствовать', 'нету', 'же', 'т', 'ли', 'по', 'о', 'тот', 'как', 'это', 'вы', 'но', 'уже'])

counter = {}

for i in range(len(messages)):
    for word in messages[i].split():
        try:
            counter[word] += messages[i].count(word)
        except:
            counter[word] = messages[i].count(word)

sort = {k: v for k, v in sorted(counter.items(), key=lambda item: item[1])}

for k in sort:
    print(sort[k], ':', k)
