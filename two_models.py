import time

from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV

from preprocess import prepr
import pandas as pd

df = pd.read_csv('data.csv')

s = time.time()
m94, y94 = prepr(df, True, labels_=[4, 9], max_words=1023622, model='ru_core_news_sm')
m, y = prepr(df, True, exclude=[4, 9], max_words=1200000, model='ru_core_news_sm')
X_main, y_main = prepr(df, True, max_words=1200000, model='ru_core_news_sm')
print('PREPROCESS: ', time.time() - s)
Tf = TfidfVectorizer()
vec = Tf.fit(m94)
X = vec.transform(m94)

Tf = TfidfVectorizer()
vec = Tf.fit(m)
X1 = vec.transform(m)

x_train, x_test, y_train, y_test = train_test_split(X, y94, test_size=0.05)
x_train1, x_test1, y_train1, y_test1 = train_test_split(X1, y, test_size=0.05)

sm = SMOTE(sampling_strategy='minority')
x_train, y_train = sm.fit_resample(x_train, y_train)

clf = LogisticRegression(max_iter=200, solver='liblinear')
clf.fit(x_train, y_train)
print(clf.score(x_test, y_test))

clf1 = LogisticRegression(max_iter=200, solver='liblinear')
clf1.fit(x_train1, y_train1)
print(clf1.score(x_test1, y_test1))

Tf = TfidfVectorizer()
vec = Tf.fit(X_main)
X_main_ = vec.transform(y_main)

x_train_m, x_test_m, y_train_m, y_test_m = train_test_split(X_main_, y_main, test_size=0.1)

eclf1 = VotingClassifier(estimators=[('lr', clf1), ('lr2', clf)], voting='hard')
eclf1 = eclf1.fit(x_train_m, y_train_m)

predictions = eclf1.predict(x_test_m)
print(f1_score(y_test_m, predictions, average='weighted'))

cm = confusion_matrix(y_test_m, predictions, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot()

plt.show()


