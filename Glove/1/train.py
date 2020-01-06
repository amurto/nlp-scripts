'''
@inproceedings{tafreshi2018emotion,
  title={Emotion Detection and Classification in a Multigenre Corpus with Joint Multi-Task Deep Learning},
  author={Tafreshi, Shabnam and Diab, Mona},
  booktitle={Proceedings of the 27th International Conference on Computational Linguistics},
  pages={2905--2913},
  year={2018}
}
'''
import os
import pickle
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import concatenate
from keras.layers.embeddings import Embedding
from keras.layers import GRU
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from collections import defaultdict
import pprint
#from keras.models import load_model


exceptions = [876, 1062, 1735, 1892, 2192, 2484, 3641, 3735, 6246, 6248, 7895, 9813, 11564, 11883, 13003, 13129, 13386, 13531, 13539, 13591, 13749, 13804, 13998, 14001, 14055, 14319, 14999]
with open('dataset.txt', encoding="utf8") as sentences:
    text_data = []
    text_labels = []
    for i,sentence in enumerate(sentences):
        if i not in exceptions:
            sentence = sentence.split('\t')
            if not sentence[2] == 'trust':                               # Remove trust labels as they are few which would cause imbalanced data
                text_data.append(sentence[1])
                text_labels.append(sentence[2])


tags = set(text_labels)
count_1 = defaultdict(int)
for item in text_labels:
    if item in tags:
        count_1[item] += 1
print('\nDataset 1 ...')
pprint.pprint(count_1)

with open('emotion_kaggle.txt', encoding="utf8") as sentences:
    kaggle_data = []
    kaggle_labels = []
    for i,sentence in enumerate(sentences):
        if i not in exceptions and i>=1:
            sentence = sentence.split(',')
            sentence[2] = sentence[2].replace('\n','')
            kaggle_data.append(sentence[1])
            kaggle_labels.append(sentence[2])

kaggle_tags = set(kaggle_labels)
count_2 = defaultdict(int)
for item in kaggle_labels:
    if item in kaggle_tags:
        count_2[item] += 1

print('\nDataset 2 ...')
pprint.pprint(count_2)

# Combine both the datasets
text_data = text_data + kaggle_data
text_labels = text_labels + kaggle_labels

count_3 = defaultdict(int)
for item in text_labels:
    count_3[item] += 1

print('\nFinal Combined Dataset ...')
pprint.pprint(count_3)

# Split dataset in training and test
X_train, X_test, y_train, y_test = train_test_split(text_data, text_labels, test_size=0.1, random_state=0)

EMBEDDING_DIMENSION = 100
MAX_SENTENCE_LENGTH = 70

# One-hot-encoding the labels
encoder = LabelEncoder()
encoder.fit(y_train)
encoded_labels_train = encoder.transform(y_train)
labels_train = to_categorical(encoded_labels_train)

# Tokenize the texts
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
sequences = tokenizer.texts_to_sequences(X_train)

word_index_data = tokenizer.word_index

# Padding to make all the texts of same length
final_data = pad_sequences(sequences, maxlen=MAX_SENTENCE_LENGTH)

# Get each word of Glove embeddings in a dictionary
embeddings_index = {}
with open(os.path.join('glove.6B.100d.txt')) as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))

# Generate Embedding Matrix from the word vectors above
embedding_matrix = np.zeros((len(word_index_data) + 1, EMBEDDING_DIMENSION))
for word, i in word_index_data.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# Both the tasks should have same dimensions of inputs
if(len(labels_train) % 2 != 0):
    labels_train = labels_train[:-1]
    final_data = final_data[:-1]

# Task-1 training data and labels
t1_y_train = labels_train[0:int(len(labels_train)/2)]
t1_x_train = final_data[0:int(len(labels_train)/2)]

# Task-2 training data and labels
t2_y_train = labels_train[int(len(labels_train)/2):]
t2_x_train = final_data[int(len(labels_train)/2):]

t1_input_layer = Input(shape=(MAX_SENTENCE_LENGTH,))
t2_input_layer = Input(shape=(MAX_SENTENCE_LENGTH,))

shared_emb_layer = Embedding(len(word_index_data) + 1, EMBEDDING_DIMENSION, weights=[embedding_matrix] , input_length=MAX_SENTENCE_LENGTH, trainable=True)
t1_emb_layer = shared_emb_layer(t1_input_layer)
t2_emb_layer = shared_emb_layer(t2_input_layer)

shared_grnn_layer = GRU(MAX_SENTENCE_LENGTH, activation='relu')
t1_grnn_layer = shared_grnn_layer(t1_emb_layer)
t2_grnn_layer = shared_grnn_layer(t2_emb_layer)

# Merging layers
merge_layer = concatenate([t1_grnn_layer, t2_grnn_layer], axis=-1)

# Task-1 Specified Layers
t1_dense_1 = Dense(30, activation='relu')(merge_layer)
t1_dropout_layer = Dropout(0.3)(t1_dense_1)
t1_dense_2 = Dense(30, activation='relu')(t1_dropout_layer)
t1_dense_3 = Dense(30, activation='relu')(t1_dense_2)
t1_prediction = Dense(labels_train.shape[1], activation='softmax')(t1_dense_3)

# Task-2 Specified Layers
t2_dense_1 = Dense(20, activation='relu')(merge_layer)
t2_dropout_layer = Dropout(0.3)(t2_dense_1)
t2_dense_2 = Dense(20, activation='relu')(t2_dropout_layer)
t2_dense_3 = Dense(20, activation='relu')(t2_dense_2)
t2_prediction = Dense(labels_train.shape[1], activation='softmax')(t2_dense_3)

# Build the model
multitask_model = Model(inputs=[t1_input_layer, t2_input_layer], outputs=[t1_prediction, t2_prediction])
# Compile the model
multitask_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001, clipvalue=1.0), metrics=['accuracy'])

# Fitting the model
multitask_model.fit([t1_x_train, t2_x_train], [t1_y_train, t2_y_train], epochs=3, batch_size=128)

# saving the model
multitask_model.save("multitask_model.h5")

# saving the tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('Model and Tokenizer saved....')

# Testing 

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
