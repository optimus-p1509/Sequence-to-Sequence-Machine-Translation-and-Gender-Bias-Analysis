
# coding: utf-8

# In[1]:


import pandas as pd
import re
import matplotlib.pyplot as plt


# In[2]:


def plot_dev_bleu():
    pattern = r'step\s(?P<step>\d+)\s.*bleu\s(?P<score>\d+\.\d+),'
    regex = re.compile(pattern)
    
    with open('2layer', 'r') as f:
        text = f.read()
    
    with open('4layer', 'r') as f:
        text2 = f.read()

    matches = regex.finditer(text)
    matches2 = regex.finditer(text2)
    
    two_layer_df = pd.DataFrame(columns=['Step', '2 Layer BLEU Score'])
    four_layer_df = pd.DataFrame(columns=['Step', '4 Layer BLEU Score'])
    
    for match in matches:
        row = pd.DataFrame([[int(match.group('step')), float(match.group('score'))]], columns=['Step', '2 Layer BLEU Score'])
        two_layer_df = pd.concat([two_layer_df, row])
    
    for match in matches2:
        row = pd.DataFrame([[int(match.group('step')), float(match.group('score'))]], columns=['Step', '4 Layer BLEU Score'])
        four_layer_df = pd.concat([four_layer_df, row])
    
    combined = two_layer_df.merge(four_layer_df, on='Step')
    
    return combined.plot(x='Step', y=['2 Layer BLEU Score', '4 Layer BLEU Score'])


# In[3]:


plot_dev_bleu()


# In[4]:


def plot_test_bleu():
    l2_df = pd.read_csv('run_.-tag-test_bleu_l2.csv')
    l4_df = pd.read_csv('run_.-tag-test_bleu_l4.csv')
    combined = l2_df.merge(l4_df, on='Step')
    combined = combined.drop(['Wall time_x', 'Wall time_y'], axis=1)
    combined.columns = ['Step', '2 Layer BLEU Score', '4 Layer BLEU Score']
    return combined.plot(x='Step', y=['2 Layer BLEU Score', '4 Layer BLEU Score'])


# In[5]:


plot_test_bleu()


# In[6]:


def plot_train_loss():
    l2_df = pd.read_csv('run_.-tag-train_loss_l2.csv')
    l4_df = pd.read_csv('run_.-tag-train_loss_l4.csv')
    dfs = [l2_df, l4_df]
    i = 0
    
    for df in dfs:
        df = df.drop(['Wall time'], axis=1)
        df.columns = ['Step', 'Train Loss']
        yield df.plot(x='Step', y='Train Loss')


# In[7]:


for p in plot_train_loss():
    p

