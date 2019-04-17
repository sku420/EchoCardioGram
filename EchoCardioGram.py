#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

#print(df.shape)
df = pd.read_csv("echocardiogram.csv")
df.head()

median = df['age'].median()
df['age'].fillna(median, inplace=True)

median = df['fractionalshortening'].median()
df['fractionalshortening'].fillna(median, inplace=True)

median = df['epss'].median()
df['epss'].fillna(median, inplace=True)

dfmedian = df['wallmotion-score'].median()
df['wallmotion-score'].fillna(median, inplace=True)

dfmedian = df['lvdd'].median()
df['lvdd'].fillna(median, inplace=True)

dfmedian = df['mult'].median()
df['mult'].fillna(median, inplace=True)

for i in df.columns:
    print(df[i].unique(),"\t",df[i].nunique())

median = df['aliveat1'].median()
df['aliveat1'].fillna(median, inplace=True)

median = df['group'].median()
df['group'].fillna(median, inplace=True)

le = LabelEncoder()
for i in df.columns:
    df[i] = le.fit_transform(df[i])

X = df.iloc[:,:12].values
y = df.iloc[:,12].values

plt.plot(X)
plt.plot(y)
plt.show()

X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.2)

#DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
print(clf.predict(X_test))
clf.score(X_test, y_test)

#RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 3)
rf.fit(X_train, y_train)
print(rf.predict(X_test))
rf.score(X_test, y_test)

#svm
sv = svm.SVC(kernel = 'linear',C=1)
sv.fit(X_train, y_train)
print(sv.predict(X_test))
sv.score(X_test, y_test)


kf = KFold(n_splits=3)
def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

folds = StratifiedKFold(n_splits=3)
scores_logistic = []
scores_svm = []
scores_rf = []
 
for train_index, test_index in kf.split(X):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index],y[train_index], y[test_index]
    scores_logistic.append(get_score(LogisticRegression(), X_train, X_test, y_train, y_test))  
    scores_svm.append(get_score(svm.SVC(), X_train, X_test, y_train, y_test))
    scores_rf.append(get_score(RandomForestClassifier(n_estimators=40), X_train, X_test, y_train, y_test))

print(scores_rf)
print(scores_svm)
print(scores_logistic)

#hyperparameter tuning
#para = { 'learning_rate' : ['constant', 'invscaling', 'adaptive'],\
        #'activation' : ['identity', 'logistic', 'tanh', 'relu'], 'solver' : ['lbfgs', 'sgd', 'adam'] }
#grid = GridSearchCV(MLPClassifier(), para,  refit = True, verbose = 0)

#decisiontree
d=np.arange(1,10)
n=np.arange(1,10)
para={'max_depth':d,'n_estimators':n}
grid= GridSearchCV(RandomForestClassifier(), para, refit = True, verbose = 0)

#svm
'''cvalues= np.arange(1,10)
gammavalues= [0.001, 0.01, 0.1, 1]
para = {'C': cvalues, 'gamma' : gammavalues}
grid= GridSearchCV(svm.SVC(kernel='linear'), para,refit = True, verbose = 0)'''
                                                                                   
print(grid.fit(X_train, y_train))
print("Best param %s"%grid.best_params_)
print("Top SCORE FOUND {}".format(grid.best_score_))
