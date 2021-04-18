#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

x = np.arange(10).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])


# In[4]:


print(x,y)


# In[5]:


model.fit(x, y)
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='warn', n_jobs=None, penalty='l2',
                   random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                   warm_start=False)
print (model.classes_)


# In[6]:


model = LogisticRegression(solver='liblinear', random_state=0)
model.fit(x, y)
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='warn', n_jobs=None, penalty='l2',
                   random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                   warm_start=False)
print (model.classes_)


# In[7]:


print(model.intercept_)


# In[8]:


print(model.predict_proba(x)
)


# In[2]:


import spacy
text = """
Dave watched as the forest burned up on the hill,
only a few miles from his house. The car had
been hastily packed and Marta was inside trying to round
up the last of the pets. "Where could she be?" he wondered
as he continued to wait for Marta to appear with the pets.
"""
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
token_list = [token for token in doc]
print(token_list)


# In[3]:


import spacy
text = """
Dave watched as the forest burned up on the hill,
only a few miles from his house. The car had
been hastily packed and Marta was inside trying to round
up the last of the pets. "Where could she be?" he wondered
as he continued to wait for Marta to appear with the pets.
"""
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
token_list = [token for token in doc]
print(token_list)


# In[4]:


filtered_tokens = [token for token in doc if not token.is_stop]
print(filtered_tokens)


# In[5]:


print(filtered_tokens[1].vector)


# In[ ]:




