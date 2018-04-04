#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 12:44:07 2018

@author: dan
"""

import numpy as np
import pandas as pd
df = pd.read_csv("/users/dan/desktop/d/t/data.csv")

df['default'].describe()
sum(df['default'] == 0)
sum(df['default'] == 1)

X = df.iloc[:, 1:6].values
y = df['default'].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, 
                                                    random_state=0)

shuffle_index = np.random.permutation(len(X_train))
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Machine Learning Algorithm
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=0)
clf.fit(X_train, y_train)

# Cross Validation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
cross_val_score(clf, X_train, y_train, cv=3, scoring='accuracy')
y_train_pred = cross_val_predict(clf, X_train, y_train, cv=3)
cm = confusion_matrix(y_train, y_train_pred)
print(cm)

from sklearn.metrics import precision_score, recall_score
print("precision score = {0:.4f}".format(precision_score(y_train, y_train_pred)))
print("recall score =  {0:.4f}".format(recall_score(y_train, y_train_pred)))

# Predicting the Test set results
y_pred = clf.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error, r2_score, roc_auc_score
print("precision score = {0:.4f}".format(precision_score(y_train, y_train_pred)))
print("recall score =  {0:.4f}".format(recall_score(y_train, y_train_pred)))
print("F1 score =  {0:.4f}".format(f1_score(y_train, y_train_pred)))
print("mean_squared_error =  {0:.4f}".format(mean_squared_error(y_train, y_train_pred)))
print("roc_auc_score=  {0:.4f}".format(roc_auc_score(y_train, y_train_pred)))

print(clf.feature_importances_)
