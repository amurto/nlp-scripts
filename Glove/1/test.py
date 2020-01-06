'''
Testing Script
'''
import json
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

MAX_SENTENCE_LENGTH = 70

exceptions = [876, 1062, 1735, 1892, 2192, 2484, 3641, 3735, 6246, 6248, 7895, 9813, 11564, 11883, 13003, 13129, 13386, 13531, 13539, 13591, 13749, 13804, 13998, 14001, 14055, 14319, 14999]
with open('dataset.txt', encoding="utf8") as sentences:
    text_data = []
    text_labels = []
    for i,sentence in enumerate(sentences):
        if i not in exceptions:
            sentence = sentence.split('\t')
            if not sentence[2] == 'trust': 
                text_data.append(sentence[1])
                text_labels.append(sentence[2])

with open('emotion_kaggle.txt', encoding="utf8") as sentences:
    kaggle_data = []
    kaggle_labels = []
    for i,sentence in enumerate(sentences):
        if i not in exceptions and i>=1:
            sentence = sentence.split(',')
            sentence[2] = sentence[2].replace('\n','')
            kaggle_data.append(sentence[1])
            kaggle_labels.append(sentence[2])


# Combine both the datasets
text_data = text_data + kaggle_data
text_labels = text_labels + kaggle_labels

# Split dataset in training and test.
X_train, X_test, y_train, y_test = train_test_split(text_data, text_labels, test_size=0.1, random_state=0)

# initialize the encoder
encoder = LabelEncoder()
encoder.fit(y_train)

# loading tokenizer for polarity model
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# loading the polarity model
multitask_model = load_model('multitask_model.h5')

# Tokenize
sequences_test = tokenizer.texts_to_sequences(X_test)

# Padding to make all the texts of same length
test_data = pad_sequences(sequences_test, maxlen=MAX_SENTENCE_LENGTH)

# Both the tasks should have same dimensions of inputs
if(len(y_test) % 2 != 0):
    y_test = y_test[:-1]
    test_data = test_data[:-1]

# Task-1 training data and labels
t1_y_test = y_test[0:int(len(y_test)/2)]
t1_x_test = test_data[0:int(len(y_test)/2)]

# Task-2 training data and labels
t2_y_test = y_test[int(len(y_test)/2):]
t2_x_test = test_data[int(len(y_test)/2):]

y_pred_combined = multitask_model.predict([t1_x_test, t2_x_test])
t1_y_pred = np.argmax(y_pred_combined[0], axis=-1)
t2_y_pred = np.argmax(y_pred_combined[1], axis=-1)

t1_y_pred = encoder.inverse_transform(t1_y_pred)
t2_y_pred = encoder.inverse_transform(t2_y_pred)

t1_acc = metrics.accuracy_score(t1_y_test, t1_y_pred)
print(f"Task 1 Accuracy: {t1_acc}")

t2_acc = metrics.accuracy_score(t2_y_test, t2_y_pred)
print(f"Task 2 Accuracy: {t2_acc}")

t1_cf = metrics.confusion_matrix(t1_y_test, t1_y_pred)
print(f'The confusion matrix for Task 1 is: \n {t1_cf}')

t2_cf = metrics.confusion_matrix(t2_y_test, t2_y_pred)
print(f'The confusion matrix for Task 2 is: \n {t2_cf}')

while True:
    predictions = {}
    query1 = str(input('Please enter the text: '))
    
    # Tokenize
    sequences_test = tokenizer.texts_to_sequences([query1])
    
    # Padding to make all the texts of same length
    test_data = pad_sequences(sequences_test, maxlen=MAX_SENTENCE_LENGTH)
    
    t1_x_test = test_data
    t2_x_test = test_data
    
    y_pred_combined = multitask_model.predict([t1_x_test, t2_x_test])
    y_pred_proba = y_pred_combined[0]
    y_pred = np.argmax(y_pred_proba, axis=-1)
    y_pred = encoder.inverse_transform(y_pred)
    
    print(f"Prediction :{y_pred}")
    
    predictions['Text Input'] = query1
    predictions['Probabilities'] = y_pred_proba.tolist()
    predictions['Label'] = y_pred.tolist()
    
    with open('result.json', 'a') as fp:
        json.dump(predictions, fp, indent=4)
