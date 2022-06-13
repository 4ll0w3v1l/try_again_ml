from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from preprocess import prepr
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

from try_again_ml import preprocess1

messages, y = prepr(pd.read_csv('data.csv'), True)

print(messages[-1], y[-1])

Tf = TfidfVectorizer( sublinear_tf=True)
vec = Tf.fit(messages)
X = vec.transform(messages)

Tfuck = vec.get_feature_names_out()

# SGD = SGDClassifier()
# parameters = {'loss': ('hinge', 'log_loss', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'),
#               'penalty': ('l2', 'l1', 'elasticnet'),
#               'alpha': (0.0001, 0.00001)}
#
# clf = GridSearchCV(SGD, parameters, verbose=2)
# clf.fit(X, y)
#
# print(clf.best_params_)

model = SGDClassifier(loss='hinge', penalty='elasticnet')

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# oversample = SMOTE(sampling_strategy='minority')
# x_train, y_train = oversample.fit_resample(x_train, y_train)

model.fit(x_train, y_train)
print(model.score(x_test, y_test))
y_pred, y_true = model.predict(x_test), y_test
print(f1_score(y_true, y_pred, average='weighted'))
# disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
# disp.plot()
# plt.show()

counter = {}

predictions = model.predict(x_test)
for inp, prediction, label in zip(x_test, predictions, y_test):
    if prediction != label:
        for line in vec.inverse_transform(inp):
            for word in line:
                try:
                    counter[word] += list(line).count(word)
                except:
                    counter[word] = list(line).count(word)

sort = {k: v for k, v in sorted(counter.items(), key=lambda item: item[1])}

counter1 = {}

predictions = model.predict(x_test)
for inp, prediction, label in zip(x_test, predictions, y_test):
    if prediction == label:
        for line in vec.inverse_transform(inp):
            for word in line:
                try:
                    counter1[word] += list(line).count(word)
                except:
                    counter1[word] = list(line).count(word)

sort1 = {k: v for k, v in sorted(counter1.items(), key=lambda item: item[1])}

max_ = max(len(sort1), len(sort1))
for i in range(max_):
    print(sort[sort.keys()[i]])

# noise = ['а', 'и', 'я', 'не', 'в', 'с', 'у', 'на', 'что', 'здравствовать', 'нету', 'же', 'т', 'ли', 'по', 'о', 'тот', 'как', 'это', 'вы', 'но', 'уже']
