# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 08:09:08 2021

@author: adgbh
"""

"""

Import Libraries

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
"""

Create a dataset

"""
dataset = pd.read_csv('exoTrain.csv')
x_train = dataset.iloc[:, 1:].values
y_train = dataset.iloc[:, 0].values
dataset_test = pd.read_csv('exoTest.csv')
x_test = dataset_test.iloc[:, 1:].values
y_test = dataset_test.iloc[:, 0].values
x_train = preprocessing.normalize(x_train)
x_test = preprocessing.normalize(x_test)

"""

Catagorical data

"""
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)
print(y_train)
"""

Plotting heatmap of missing values

"""
sns.heatmap(dataset.isnull())
sns.heatmap(dataset.corr(), fmt = ".2g", cmap = "Blues_r", linewidths =".1")
"""

Plotting Gaussian histograms

"""
labels_1=[100,200,300] # Non Exoplanet Data
for i in labels_1:
    plt.figure(figsize=(3,3))
    plt.hist(dataset.iloc[i,:], bins=200)
    plt.title("gaussian histogram")
    plt.xlabel("flux values")
    plt.show()
    
labels_1=[13,25,28] # Exoplanet data
for i in labels_1:
    plt.figure(figsize=(3,3))
    plt.hist(dataset.iloc[i,:], bins=200)
    plt.title("gaussian histogram")
    plt.xlabel("flux values")
    plt.show()
"""

Box plot of flux to find outliers

"""
fig, axes = plt.subplots (1, 5, figsize = (15,6), sharey = True)
fig.suptitle('Flux Distribution')

sns.boxplot(ax = axes[0], data=dataset, x='LABEL', y='FLUX.1', palette='Set2')
sns.boxplot(ax = axes[1], data=dataset, x='LABEL', y='FLUX.2', palette='Set2')
sns.boxplot(ax = axes[2], data=dataset, x='LABEL', y='FLUX.3', palette='Set2')
sns.boxplot(ax = axes[3], data=dataset, x='LABEL', y='FLUX.4', palette='Set2')
sns.boxplot(ax = axes[4], data=dataset, x='LABEL', y='FLUX.5', palette='Set2')
"""

Removing outliers and checking in box plot

"""
dataset.drop(dataset[dataset['FLUX.1']>250000].index, axis=0, inplace = True)
dataset.drop(dataset[dataset['FLUX.1']<-200000].index, axis=0, inplace = True)
fig, axes = plt.subplots(1,5,figsize =(15,6), sharey = True)
fig.suptitle('distribution of flux')

sns.boxplot(ax = axes[0], data=dataset, x='LABEL', y='FLUX.1', palette='Set2')
sns.boxplot(ax = axes[1], data=dataset, x='LABEL', y='FLUX.2', palette='Set2')
sns.boxplot(ax = axes[2], data=dataset, x='LABEL', y='FLUX.3', palette='Set2')
sns.boxplot(ax = axes[3], data=dataset, x='LABEL', y='FLUX.4', palette='Set2')
sns.boxplot(ax = axes[4], data=dataset, x='LABEL', y='FLUX.5', palette='Set2')
"""
Decision tree Algorithm

"""
from sklearn.tree import DecisionTreeClassifier
regressor = DecisionTreeClassifier(random_state = 0)
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
print ("Accuracy of DTR")
from sklearn.metrics import accuracy_score
# print (y_pred)
print (accuracy_score(y_test,y_pred, normalize = True))
"""

Random Forest Algorithm

"""
from sklearn.ensemble import RandomForestClassifier
rf_regressor = RandomForestClassifier(n_estimators=5, random_state = 0)
rf_regressor.fit(x_train, y_train)
y_pred_rf = rf_regressor.predict(x_test)
from sklearn.metrics import accuracy_score
print ("Accuracy of RFA")
# print (accuracy_score(y_test,y_pred_rf, normalize = True))
print (y_pred_rf)

