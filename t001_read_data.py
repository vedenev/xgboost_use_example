# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 13:48:38 2019

@author: vedenev
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier

train_path = './train.csv'
#train_path = './test.csv'

df = pd.read_csv(train_path, header=None)

labels = df[df.columns[-1]].to_numpy()
#print(np.unique(labels)) # [0 1 2 3] # 4 differnet labels
#np.any(np.isnan(labels)): False for train, False for test

labels_hist = np.histogram(labels, bins=[-0.5, 0.5, 1.5, 2.5, 3.5])
# Trian:(array([2177, 2152, 2202, 2147], dtype=int64) # balanced dataset
# Test: (array([712, 777, 777, 734], dtype=int64) # balanced dataset

categorial_feature = df[df.columns[-2]].tolist()
#print(list(set(categorial_feature))) # ['B', 'C', 'A', 'D'] # 4 categies

numericle_features = df[df.columns[0:-2]].to_numpy()
#print(np.min(numericle_features)) # -128
#print(np.max(numericle_features)) # 127
#numericle_features origninaly was int8 type 

# by looking on the train csv file see correlation between categorial feature and label:
# C,0
# A,1
# D,2
# B,3
# so let convert categorial from leters to numbers:
dataset_size = len(categorial_feature)
categorial_feature_numbers = np.zeros(dataset_size, np.int64)
convertion_dict = {'C': 0, 'A': 1, 'D': 2, 'B': 3}
for sample_count in range(dataset_size):
    categorial_feature_numbers[sample_count] = convertion_dict[categorial_feature[sample_count]]

#print(np.all(categorial_feature_numbers == labels)) # True in train, False in test
R = np.corrcoef(categorial_feature_numbers.astype(np.float32), labels.astype(np.float32))
#R[0, 1] = 1.0 # for train
#R[0, 1] = 0.00563114 # for test
# so categorial feature needs to though away (because in train it just repeat labels, so no aditional info from it)

 
# empties analysis:
#print(np.any(np.isnan(numericle_features))) # False for train, True for test
# to print emptyes in test:
#np.where(np.isnan(numericle_features))
# result:
#(array([ 10,  13,  13,  22,  27,  29,  31, 189, 196, 197, 204, 207], dtype=int64),
# array([ 6, 17, 22,  7, 18, 17,  5, 14,  4, 17, 19, 14], dtype=int64))  
# 12 empties in test

model = XGBClassifier()
#model.fit(numericle_features, labels)

#from sklearn.model_selection import cross_val_score
#scores = cross_val_score(model, numericle_features, labels, cv=5)
#socres: array([0.92807825, 0.93268124, 0.92906574, 0.93944637, 0.92329873]) # approximatly same stable result -> all ok

model.fit(numericle_features, labels)


# model testing:

test_path = './test.csv'
df_test = pd.read_csv(train_path, header=None)

labels_test = df_test[df_test.columns[-1]].to_numpy()

categorial_feature_test = df_test[df_test.columns[-2]].tolist()

numericle_features_test = df_test[df_test.columns[0:-2]].to_numpy()

# prepare mean statisitics from train:
numericle_features_stat = np.mean(numericle_features, axis=0)

# replace empties in test with mean values:
empty_inds_sample_no, empty_inds_feature_no = np.where(np.isnan(numericle_features_test))
numericle_features_test_filled = np.copy(numericle_features_test)
for empty_count in range(empty_inds_sample_no.size):
    numericle_features_test_filled[empty_inds_sample_no[empty_count], empty_inds_feature_no[empty_count]] = numericle_features_stat[empty_inds_feature_no[empty_count]]

prediction_test = model.predict(numericle_features_test_filled)

from sklearn.metrics import accuracy_score
accuracy_test = accuracy_score(labels_test, prediction_test)
#accuracy_test: 0.9613966351693939
    
