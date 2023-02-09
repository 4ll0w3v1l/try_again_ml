from string import punctuation

from nltk.corpus import stopwords
from pymystem3 import Mystem
import re

mystem = Mystem()
russian_stopwords = stopwords.words("russian")


def prepr(dataset, isPd=None):
    if isPd:
        df = dataset.dropna()

        labels = []
        full_text = ''

        for i in range(len(df.iloc[:, 0])):
            try:
                message = re.sub('[^а-яё]', ' ', df.iloc[i, 0].lower()).split(' ')
                for word in message:
                    if word != '' and word not in russian_stopwords:
                        full_text += word + ' '
                full_text += '/'
                labels.append(df.iloc[i, 1])
            except Exception as e:
                print(e)

        Prepr = mystem.lemmatize(full_text)
        text = ''.join(Prepr)

        return text.split('/')[0:-1], labels;

    else:
        tokens = mystem.lemmatize(re.sub('[^а-я a-zА-Я]', '', dataset.lower()))
        tokens = [token for token in tokens if token not in russian_stopwords \
                  and token != " " \
                  and token.strip() not in punctuation]

        text = " ".join(tokens)

        return text


