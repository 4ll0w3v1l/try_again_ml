from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV

from preprocess import prepr
import pandas as pd

df = pd.read_csv('data.csv')

m94, y94 = prepr(df, True, labels_=[4, 9], max_words=1200000)
m, y = prepr(df, True, exclude=[4, 9], max_words=1200000)

Tf = TfidfVectorizer()
vec = Tf.fit(m94)
X = vec.transform(m94)

Tf = TfidfVectorizer()
vec = Tf.fit(m)
X1 = vec.transform(m)

x_train, x_test, y_train, y_test = train_test_split(X, y94, test_size=0.1)
x_train1, x_test1, y_train1, y_test1 = train_test_split(X1, y, test_size=0.1)

clf = LogisticRegression(max_iter= 200,solver='liblinear')
clf.fit(x_train, y_train)
print(clf.score(x_test, y_test))

clf1 = LogisticRegression(max_iter= 200,solver='liblinear')
clf1.fit(x_train1, y_train1)
print(clf1.score(x_test1, y_test1))
