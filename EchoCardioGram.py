#!/usr/bin/env python
# coding: utf-8

# In[3]:


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


# In[4]:


#print(df.shape)
df = pd.read_csv("echocardiogram.csv")
df.head()


# In[5]:


median = df['age'].median()
df['age'].fillna(median, inplace=True)


# In[6]:


median = df['fractionalshortening'].median()
df['fractionalshortening'].fillna(median, inplace=True)


# In[7]:


median = df['epss'].median()
df['epss'].fillna(median, inplace=True)


# In[8]:


dfmedian = df['wallmotion-score'].median()
df['wallmotion-score'].fillna(median, inplace=True)


# In[9]:


dfmedian = df['lvdd'].median()
df['lvdd'].fillna(median, inplace=True)


# In[10]:


dfmedian = df['mult'].median()
df['mult'].fillna(median, inplace=True)


# In[11]:


for i in df.columns:
    print(df[i].unique(),"\t",df[i].nunique())


# In[12]:


median = df['aliveat1'].median()
df['aliveat1'].fillna(median, inplace=True)


# In[13]:


median = df['group'].median()
df['group'].fillna(median, inplace=True)


# In[14]:


le = LabelEncoder()
for i in df.columns:
    df[i] = le.fit_transform(df[i])


# In[15]:


X = df.iloc[:,:12].values
y = df.iloc[:,12].values


# In[16]:


plt.plot(X)
plt.plot(y)
plt.show()


# In[27]:


X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.2)


# In[28]:


clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)


# In[29]:


print(clf.predict(X_test))
clf.score(X_test, y_test)


# In[30]:


rf = RandomForestClassifier(n_estimators = 3)
rf.fit(X_train, y_train)


# In[31]:


print(rf.predict(X_test))
rf.score(X_test, y_test)


# In[32]:


sv = svm.SVC(kernel = 'linear',C=1)
sv.fit(X_train, y_train)


# In[33]:


print(sv.predict(X_test))
sv.score(X_test, y_test)


# In[34]:


kf = KFold(n_splits=3)


# In[35]:


def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


# In[36]:


folds = StratifiedKFold(n_splits=3)
scores_logistic = []
scores_svm = []
scores_rf = []
 
for train_index, test_index in kf.split(X):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index],                                        y[train_index], y[test_index]
    scores_logistic.append(get_score(LogisticRegression(), X_train, X_test, y_train, y_test))  
    scores_svm.append(get_score(svm.SVC(), X_train, X_test, y_train, y_test))
    scores_rf.append(get_score(RandomForestClassifier(n_estimators=40), X_train, X_test, y_train, y_test))


# In[37]:


scores_rf


# In[38]:


scores_svm


# In[39]:


scores_logistic


# In[40]:


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
                                          


# In[41]:


print("Best param %s"%grid.best_params_)
print("Top SCORE FOUND {}".format(grid.best_score_))


# In[ ]:




