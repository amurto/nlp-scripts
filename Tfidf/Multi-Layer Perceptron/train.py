#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegression


df=pd.read_csv("ner_dataset.csv",encoding='latin1')
df = df.fillna(method='ffill')
df

import nltk
nltk.download('punkt')

from sklearn.model_selection import train_test_split

x_df=df['Word']
y_df=df['Tag']
X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, random_state=0)

from sklearn.feature_extraction.text import TfidfVectorizer

# Fit the TfidfVectorizer to the training data specifiying a minimum document frequency of 5
vect = TfidfVectorizer(min_df=5).fit(X_train)
len(vect.get_feature_names())
X_train_vectorized = vect.transform(X_train)

from sklearn.metrics import roc_auc_score
from sklearn import preprocessing

def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)

model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

predictions = model.predict(vect.transform(X_test))

print('AUC: ', multiclass_roc_auc_score(y_test, predictions))

import pickle

filename = 'Log_model.sav'
pickle.dump(model, open(filename, 'wb'))

#Importing MLPClassifier
from sklearn.neural_network import MLPClassifier

#Initializing the MLPClassifier
classifier = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=100,activation = 'relu',solver='adam',random_state=1)

def accuracy(confusion_matrix):
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements

X_train_vectorized = vect.transform(X_train)

from sklearn.metrics import roc_auc_score
from sklearn import preprocessing

def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)

#model = LogisticRegression()
classifier.fit(X_train_vectorized, y_train)

predictions = classifier.predict(vect.transform(X_test))

print('AUC: ', multiclass_roc_auc_score(y_test, predictions))

#Fitting the training data to the network
#classifier.fit(X_train, Y_train)
#Predicting y for X_val
#y_pred = classifier.predict(X_val)

import pickle

filename2 = 'mlp_model.sav'
pickle.dump(classifier, open(filename2, 'wb'))

