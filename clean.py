# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 17:25:46 2018

@author: andyj
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
import matplotlib.pylab as plt

# import
df_test = pd.read_csv("data/test.csv")
df_test_labs = pd.read_csv("data/test_labels.csv")
df_train = pd.read_csv("data/train.csv")
df_sub = pd.read_csv("data/sample_submission.csv")


#parse and clean comments
comment = re.sub('[^a-zA-Z]', ' ', df_train['comment_text'][0]).lower()