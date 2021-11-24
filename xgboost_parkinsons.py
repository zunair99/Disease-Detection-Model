#importing all necessary packages
from itertools import count
import numpy as np
import pandas as pd
import os, sys
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#importing .data file, creating dataframe from it, and getting a basic understanding of what the data is comprised of
df = pd.read_csv('Detecting-Parkinsons-Disease-master\Detecting-Parkinsons-Disease-master\parkinsons.data')
print(df)
df.head()
df.columns
df.tail

#creating features and labels, features represented by all columns except for status (status is the label for each data value)
features=df.loc[:,df.columns!='status'].values[:,1:]
features
labels=df.loc[:,'status'].values
labels

#determining how many status values are 0 and how many are 1 in the array labels
labels_one = np.count_nonzero(labels == 1)
print(labels_one)
labels_zero = np.count_nonzero(labels == 0)
print(labels_zero)

#implementation of scikit-learn package to use MinMaxScaler estimator
estimator = MinMaxScaler((-1,1))
x = estimator.fit_transform(features)
x
y = labels
y

#splitting dataset into training and testing datasets (20% for testing)
x.shape
y.shape
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#use of XGBClassifier to train the model
classifier = XGBClassifier()
classifier.fit(x_train,y_train)

#predict values for x_test, calculating accuracy of the model
y_prediction = classifier.predict(x_test)
print(accuracy_score(y_test,y_prediction)*100)
#84.62% accurate