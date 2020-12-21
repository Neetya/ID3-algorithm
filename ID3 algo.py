
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import copy
df = pd.read_csv("Admission_Predict.csv")
df.head()


# In[2]:


df.columns = ['Serial No.', 'GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research', 'Chance of Admit']
df_modified = pd.DataFrame()


# In[3]:


# adjusting GRE scores
temp = []
[ temp.append('GRE_Good') if el >= 310 else temp.append('GRE_Poor') if el <= 290 else temp.append('GRE_Med') for el in df['GRE Score'] ]
df_modified['GRE Score'] = temp

# adjusting toefl score
temp = []
[ temp.append('TOEFL_Good') if el >= 110 else temp.append('TOEFL_Poor') for el in df['TOEFL Score'] ]
df_modified['TOEFL Score'] = temp

# adjusting univ rating
temp = []
[ temp.append('UNI_Good') if el >= 4.0 else temp.append('UNI_Poor') for el in df['University Rating'] ]
df_modified['University Rating'] = temp

# adjusting SOP
temp = []
[ temp.append('SOP_Good') if el >= 4.0 else temp.append('SOP_Poor') for el in df['SOP'] ]
df_modified['SOP'] = temp

# adjusting LOR
temp = []
[ temp.append('LOR_Good') if el >= 4.0 else temp.append('LOR_Poor')for el in df['LOR'] ]
df_modified['LOR'] = temp

# adjusting CGPA
temp = []
[ temp.append('CGPA_Good') if el >= 8.5 else temp.append('CGPA_Poor') if el <= 7 else temp.append('CGPA_Med') for el in df['CGPA'] ]
df_modified['CGPA'] = temp

# adjusting research
temp = []
[ temp.append('Research_Yes') if el == 1 else temp.append('Research_No') for el in df['Research'] ]
df_modified['Research'] = temp

# adjusting chance of admit
temp = []
[ temp.append('Admit_Yes') if el >= 0.5 else temp.append('Admit_No') for el in df['Chance of Admit'] ]
df_modified['Chance of Admit'] = temp


# In[4]:


X = df_modified.iloc[:, :].values
attribute = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research']
r = 400
c = 7
df_modified.head()


# In[5]:


class Node(object):
    def __init__(self):
        self.value = None
        self.decision = None
        self.childs = None
        
def findEntropy(data, rows):
    yes = 0
    no = 0
    ans = -1
    idx = len(data[0]) - 1
    entropy = 0
    for i in rows:
        if data[i][idx] == 'Admit_Yes':
            yes = yes + 1
        else:
            no = no + 1

    x = yes/(yes+no)
    y = no/(yes+no)
    if x != 0 and y != 0:
        entropy = -1 * (x*math.log2(x) + y*math.log2(y))
    if x == 1:
        ans = 1
    if y == 1:
        ans = 0
    return entropy, ans

def findMaxGain(data, rows, columns):
    maxGain = 0
    retidx = -1
    entropy, ans = findEntropy(data, rows)
    if entropy == 0:
        return maxGain, retidx, ans

    for j in columns:
        mydict = {}
        idx = j
        for i in rows:
            key = data[i][idx]
            if key not in mydict:
                mydict[key] = 1
            else:
                mydict[key] = mydict[key] + 1
        gain = entropy

        for key in mydict:
            yes = 0
            no = 0
            for k in rows:
                if data[k][j] == key:
                    if data[k][-1] == 'Admit_Yes':
                        yes = yes + 1
                    else:
                        no = no + 1
            x = yes/(yes+no)
            y = no/(yes+no)
            if x != 0 and y != 0:
                gain += (mydict[key] * (x*math.log2(x) + y*math.log2(y)))/r
        if gain > maxGain:
            maxGain = gain
            retidx = j

    return maxGain, retidx, ans

def buildTree(data, rows, columns):

    maxGain, idx, ans = findMaxGain(X, rows, columns)
    root = Node()
    root.childs = []
    if maxGain == 0:
        if ans == 1:
            root.value = 'Admit_Yes'
        else:
            root.value = 'Admit_No'
        return root
    root.value = attribute[idx]
    mydict = {}
    for i in rows:
        key = data[i][idx]
        if key not in mydict:
            mydict[key] = 1
        else:
            mydict[key] += 1
    newcolumns = copy.deepcopy(columns)
    newcolumns.remove(idx)
    for key in mydict:
        newrows = []
        for i in rows:
            if data[i][idx] == key:
                newrows.append(i)
        temp = buildTree(data, newrows, newcolumns)
        temp.decision = key
        root.childs.append(temp)
    return root

def traverse(root):
    print(root.decision)
    print(root.value)

    n = len(root.childs)
    if n > 0:
        for i in range(0, n):
            traverse(root.childs[i])
            
def calculate():
    rows = [i for i in range(0, r)]
    columns = [i for i in range(0, c)]
    root = buildTree(X, rows, columns)
    root.decision = 'Start'
    traverse(root)

calculate()

