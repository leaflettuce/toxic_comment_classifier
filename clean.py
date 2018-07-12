# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 17:25:46 2018

@author: andyj
"""
import pandas as pd
import numpy as np
import re
import os
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#set working dir
os.chdir('D:/Projects/Kaggle/toxic_comments')

# import
df_test = pd.read_csv("data/test.csv")
df_test_labs = pd.read_csv("data/test_labels.csv")
df_train = pd.read_csv("data/train.csv")
df_sub = pd.read_csv("data/sample_submission.csv")

#init corpus
corpus = []

#loop through df and clean comments
for i in range(0, 5000):
    #reg_exp to replace anything not text to a space and drop to lower case
    comment = re.sub('[^a-zA-Z]', ' ', df_train['comment_text'][i]).lower()
    #split into list for processing
    comment = comment.split()
    #check for stopwords and remove
    comment = [word for word in comment if not word in set(stopwords.words('english'))]
    #stem the word!
    ps = PorterStemmer()
    comment = [str(ps.stem(word)) for word in comment]
    #back to string
    comment = ' '.join(comment)
    corpus.append(comment)
    #track progress
    if i%1000 == 0:
        print((float(i)/len(df_train))*100)
        
#Bag of Words Model - sparse matrix (tokenize)
cv = CountVectorizer(max_features = 10000)   #max words to store 
X = cv.fit_transform(corpus).toarray()
y_tox = df_train.iloc[0:5000,2].values
y_sev_tox = df_train.iloc[0:5000,3].values
y_obs = df_train.iloc[0:5000,4].values
y_threat = df_train.iloc[0:5000,5].values
y_insult = df_train.iloc[0:5000,6].values
y_hate = df_train.iloc[0:5000,7].values

#Dimensionality Reduction

#model for each predicted type
tests = {'y_tox' : y_tox, 
         'y_sev_tox' : y_sev_tox, 
         'y_obs' : y_obs, 
         'y_threat' : y_threat, 
         'y_insult' : y_insult, 
         'y_hate' : y_hate}

models = {'y_tox' : GaussianNB(), 
          'y_sev_tox' : GaussianNB(),
          'y_obs' : GaussianNB(),
          'y_threat' : GaussianNB(),
          'y_insult' : GaussianNB(),
          'y_hate' : GaussianNB()}

preds = {}

test_names = ['y_tox', 'y_sev_tox', 'y_obs', 'y_threat', 'y_insult', 'y_hate']
for i in test_names:
    #test_train split (toxic)
    X_train, X_test, y_train, y_test = train_test_split(X, tests[i], test_size = 0.25, random_state = 42)
    
    #Train Model (naive bayes)
    models[i].fit(X_train, y_train)

    #predict
    preds[i] = models[i].predict(X_test)
    
    #review model
    print(i)
    print(confusion_matrix(y_test, preds[i]))
    print(accuracy_score(y_test, preds[i]))


# RUN TEST THROUGH PIPELINE
test_corpus = []
for i in range(0, 5000):
    #reg_exp to replace anything not text to a space and drop to lower case
    comment = re.sub('[^a-zA-Z]', ' ', df_test['comment_text'][i]).lower()
    #split into list for processing
    comment = comment.split()
    #check for stopwords and remove
    comment = [word for word in comment if not word in set(stopwords.words('english'))]
    #stem the word!
    ps = PorterStemmer()
    comment = [str(ps.stem(word)) for word in comment]
    #back to string
    comment = ' '.join(comment)
    test_corpus.append(comment)
    #track progress
    if i%1000 == 0:
        print((float(i)/len(df_train))*100)
        
#kaggle test array    
X_kaggle = cv.transform(test_corpus).toarray()
kaggle_preds = {}

#predict probability for each
for i in test_names:
    kaggle_preds[i] = models[i].predict_proba(X_kaggle)

#list out in sample sub
sample_sub = df_sub[0:5000]
for i in range(0, 5000):
    sample_sub.iloc[i,1] = kaggle_preds['y_tox'][i][1]
    sample_sub.iloc[i,2] = kaggle_preds['y_sev_tox'][i][1]
    sample_sub.iloc[i,3] = kaggle_preds['y_obs'][i][1]
    sample_sub.iloc[i,4] = kaggle_preds['y_threat'][i][1]
    sample_sub.iloc[i,5] = kaggle_preds['y_insult'][i][1]
    sample_sub.iloc[i,6] = kaggle_preds['y_hate'][i][1]
    #track progress
    if i%500 == 0:
        print(float(i)/len(sample_sub))
    
#Write to CSV
sample_sub.to_csv('tox_nb_sub_test.csv', index=False)