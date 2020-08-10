#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import re
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


# In[ ]:


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer() 


# In[ ]:


df = pd.read_csv('../data/Reviews.csv')
df.head()


# In[ ]:


df['Text'] = df['Text'] + ' ' + df['Summary']


# In[ ]:


# Delete unused columns
del df['Id']
del df['ProfileName']
del df['Summary']
del df['HelpfulnessNumerator']
del df['HelpfulnessDenominator']
del df['Time']
del df['ProductId']


# In[ ]:


df.head()


# In[ ]:


df.loc[df['Score'] <= 3, 'ReviewSentiment'] = 0
df.loc[df['Score'] > 3, 'ReviewSentiment'] = 1

df['ReviewSentiment'] = df['ReviewSentiment'].astype(int)


# In[ ]:


df.isna().sum()


# In[ ]:


#convert na to ""
df['Text'].fillna("", inplace=True)


# In[ ]:


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
    


# In[ ]:


df['Text'] = df['Text'].apply(preprocess)


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(df['Text'], df['ReviewSentiment'], test_size=0.2, random_state=1, stratify=df['ReviewSentiment'])


# In[ ]:


print('Number of train samples:', len(x_train))
print('Number of test samples:', len(x_test))


# In[ ]:


# Delete unused objects
del stop_words
del lemmatizer
del df


# In[ ]:


tfidfv = TfidfVectorizer()
fv_train = tfidfv.fit_transform(x_train)
fv_test = tfidfv.transform(x_test)

print('Shape of train count vector:', fv_train.shape)
print('Shape of test count vector:', fv_test.shape)


# In[ ]:


lr = LogisticRegression(penalty='l2', random_state=1, solver='sag', max_iter=1000, class_weight='balanced')


# In[ ]:


parameters = {'C':[3, 4, 5, 6, 7, 8, 9]}
#parameters = {'C':[3, 4]}
print('Fitting GridSearchCV...')
grid_cv = GridSearchCV(lr, parameters)
grid_cv.fit(fv_train, y_train)
print('Completed fitting GridSearchCV.')


# In[ ]:


#lr.fit(fv_train, y_train)
train_acc = grid_cv.score(fv_train, y_train)
print('Train Accuracy: %.3f' %(train_acc))

test_acc = grid_cv.score(fv_test, y_test)
print('Test Accuracy: %0.3f' % (test_acc))


# In[ ]:


# Write model to disk
pickle.dump(grid_cv, open('model/lr_bal_grid_model', 'wb'))
loaded_grid_cv = pickle.load(open('model/lr_bal_grid_model', 'rb'))
print('Test Accuracy from Loaded Model: %.3f' % (loaded_grid_cv.score(fv_test, y_test)))
print('Best C:', loaded_grid_cv.best_params_)

