#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


# In[2]:


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


# In[3]:


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

df = pd.read_csv('../data/Reviews.csv')
df['Text'] = df['Text'] + ' ' + df['Summary']


# In[4]:


# Delete unused columns
del df['Id']
del df['ProfileName']
del df['Summary']
del df['HelpfulnessNumerator']
del df['HelpfulnessDenominator']
del df['Time']
del df['ProductId']


# In[5]:


df.head()


# In[6]:


df.loc[df['Score'] <= 3, 'ReviewSentiment'] = 0
df.loc[df['Score'] > 3, 'ReviewSentiment'] = 1

df['ReviewSentiment'] = df['ReviewSentiment'].astype(int)


# In[7]:


df.isna().sum()


# In[8]:


#convert na to ""
df['Text'].fillna("", inplace=True)


# In[9]:


def preprocess(s):
    # Remove html tags
    s = re.sub('<\S+>', '', s)
    # Replace urls with token
    s = re.sub(r'http:\S+', 'url', s)
    s = re.sub(r'https:\S+', 'url', s)
    
    s = s.lower()
    # Remove any other special characters
    s = re.sub(r'[^a-z ]', ' ', s)
    
    words = s.split()
    result = []
    
    # Remove stop words and lemmatize the words
    for word in words:
        if word in stop_words:
            continue
        word = lemmatizer.lemmatize(word)
        result.append(word)
    return ' '.join(result)

df['Text'] = df['Text'].apply(preprocess)


# In[10]:


x_train, x_test, y_train, y_test = train_test_split(df['Text'], df['ReviewSentiment'], test_size=0.2, random_state=1, stratify=df['ReviewSentiment'])
print('Number of train samples:', len(x_train))
print('Number of test samples:', len(x_test))


# In[11]:


# Delete unused objects
del stop_words
del lemmatizer
del df


# In[12]:


def doc2vec(s):
    words = word_tokenize(s)
    M = []
    for w in words:
        try:
            M.append(embeddings_index[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(300)
    return v / np.sqrt((v ** 2).sum())


# In[13]:


print(x_train[0])
print(y_train[0])


# In[14]:


xtrain_glove = [doc2vec(x) for x in tqdm(x_train)]


# In[15]:


xtest_glove = [doc2vec(x) for x in tqdm(x_test)]


# In[ ]:


print('Starting GridSearchCV Training...')
lr = LogisticRegression(penalty='l2', random_state=1, solver='sag', max_iter=1000, class_weight='balanced', verbose=1)
parameters = {'C':[2,5,8,10]}
#lr.fit(xtrain_glove, y_train.tolist())
grid_cv = GridSearchCV(lr, parameters)
grid_cv.fit(xtrain_glove, y_train.tolist())
print('GridSearchCV Training Complete.')


# In[ ]:


print('Accuracy on Test data:', grid_cv.score(xtest_glove, y_test))


# In[ ]:


pickle.dump(grid_cv, open('model/lr_glove_grid_model', 'wb'))
loaded_grid_cv = pickle.load(open('model/lr_glove_grid_model', 'rb'))
print('Loaded GridCV Model on test data:', loaded_grid_cv.score(xtrain_glove, y_train.tolist()))

