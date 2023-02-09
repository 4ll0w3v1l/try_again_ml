"""
переменная df подразумевает импортированный файл через pd.read()
"""
import pickle
import preprocess

vec = pickle.load(open('vec.pkl', mode='rb'))
model = pickle.load(open('model.pkl', mode='rb'))


def train(df):
    messages, labels = preprocess.prepr(df, isPd=True)

    X = vec.transform(messages)
    y = labels

    model.fit(X, y)

    return 'TRAINED'


def predict(inp):
    text = preprocess.prepr(inp)

    X = vec.transform([text])

    return model.predict(X)

