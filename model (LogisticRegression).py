# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 23:03:32 2018

@author: Utkarsh
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir('C:/Users/alien/Downloads/Compressed/iris-flower-dataset')

dataset = pd.read_csv("IRIS.csv")

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

for i in range(len(x)):
    if y[i] == 'Iris-virginica':
        y[i] = 1
    else:
        y[i] = 0

from sklearn.preprocessing import StandardScaler
SC = StandardScaler()
sc_X = SC.fit_transform(x)

#Split dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(sc_X, y, test_size = 0.25, random_state = 49)
y_train = y_train.astype('int')
y_test = y_test.astype('int')

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)