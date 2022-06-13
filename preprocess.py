import re
import spacy
import pandas as pd


def prepr(data, isdf=None, stop_words=[]):
    if stop_words is None:
        stop_words = []
    if isdf:
        nlp = spacy.load("ru_core_news_lg")

        df = data.dropna()

        tmp_mess = ''
        labels = []

        for i in range(len(df.iloc[:, 0])):
            try:
                int(df.iloc[i, 1])
                tmp_mess += re.sub('[^а-яё]', ' ', str(df.iloc[i, 0]).strip().lower()) + ' / '
                labels.append(df.iloc[i, 1])
            except:
                pass

        noise = stop_words

        doc = nlp(tmp_mess)
        tokens = ' '.join(token.lemma_ for token in doc if token.lemma_.strip() != '')
        messages = tokens.split('/')

        for i in range(len(messages)):
            tmp = []
            for word in messages[i].split():
                if word not in noise:
                    tmp.append(word)
            messages[i] = ' '.join(tmp)

        return messages[:-1], labels
    else:
        pass