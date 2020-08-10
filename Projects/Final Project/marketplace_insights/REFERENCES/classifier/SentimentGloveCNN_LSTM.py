#!/usr/bin/env python
# coding: utf-8

# In[4]:


import nltk
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding


# In[27]:


embeddings_index = {}
with open('glove/glove.6B.300d.txt', encoding="utf8") as f:
    line = f.readline()
    while line:
        values = line.split()
        word = values[0]
        try:
           coefs = np.asarray(values[1:], dtype='float32')
           embeddings_index[word] = coefs
        except ValueError:
           pass
        line = f.readline()


# In[12]:


df = pd.read_csv('../data/preprocessed.csv', index_col=False)
df.dropna(inplace = True) 


# In[16]:


df.head()


# In[28]:


# From Preprocess.ipynb got vocabulary_size
vocabulary_size = 103726
tokenizer = Tokenizer(num_words= vocabulary_size)
tokenizer.fit_on_texts(df['Text'])
sequences = tokenizer.texts_to_sequences(df['Text'])
data = pad_sequences(sequences, maxlen=100)


# In[24]:


labels = df['ReviewSentiment'].tolist()


# In[29]:


embedding_matrix = np.zeros((vocabulary_size, 300))
for word, index in tokenizer.word_index.items():
    if index > vocabulary_size - 1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector


# In[ ]:


model_glove = Sequential()
model_glove.add(Embedding(vocabulary_size, 300, input_length=100, weights=[embedding_matrix], trainable=False))
model_glove.add(Dropout(0.2))
model_glove.add(Conv1D(64, 5, activation='relu'))
model_glove.add(MaxPooling1D(pool_size=4))
model_glove.add(LSTM(100))
model_glove.add(Dense(1, activation='sigmoid'))
model_glove.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

class_weight = {0: 1.0,
                1: 3.5}
model_glove.fit(data, np.array(labels), validation_split=0.2, class_weight=class_weight, epochs = 10)


# In[26]:


model_glove.save('model/cnn1_lstm1_glove')


# In[ ]:




