# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: utils.py

@time: 2019/1/5 10:03

@desc:

"""

import pickle
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import sys

def pickle_load(file_path):
    return pickle.load(open(file_path, 'rb'))


def pickle_dump(obj, file_path):
    pickle.dump(obj, open(file_path, 'wb'))


def get_score_senti(y_true, y_pred):
    """
    return score for predictions made by sentiment analysis model
    :param y_true: array shaped [batch_size, 3]
    :param y_pred: array shaped [batch_size, 3]
    :return:
    """
    y_true = np.argmax(y_true, axis=-1)
    y_pred = np.argmax(y_pred, axis=-1)
    #print(y_true)
    #print('PREDICTION')
    #print(y_pred)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')

    print('acc:', acc)
    print('macro_f1:', f1)
    return acc, f1

def get_score_senti2(y_pred):
    """
    return score for predictions made by sentiment analysis model
    :param y_true: array shaped [batch_size, 3]
    :param y_pred: array shaped [batch_size, 3]
    :return:
    """
    #y_true = np.argmax(y_true, axis=-1)
    y_pred = np.argmax(y_pred, axis=-1)
    #print("The prediction is:" ,str(y_pred))
    if(y_pred==0):
        print("The aspect of the sentence is negative")
    elif(y_pred==2):
        print("The aspect of the sentence is positive")
    else:
        print("The aspect of the sentence is neutral")
    sys.exit()
    #acc = accuracy_score(y_true, y_pred)
    #f1 = f1_score(y_true, y_pred, average='macro')

