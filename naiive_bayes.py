from sklearn.naive_bayes import GaussianNB, BernoulliNB
from pandas import read_csv
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

#Import data
dataset = read_csv('data/train.csv', header=0)
target = dataset['is_female']

#Move class column to the first column
dataset.drop(labels=['is_female'], axis=1, inplace =True)
dataset.insert(0, 'is_female', target)

#Drop columns with NA
print len(dataset.columns)
dataset = dataset.dropna(axis=1)
print len(dataset.columns)

##Replace NA columns with value -1
#dataset = dataset.fillna(-1)

# TRAIN AND TEST ON TRAINING SET
# Split training set to train and test
x_train, x_test, y_train, y_test = train_test_split(dataset, target, test_size=0.4, random_state=0)

# Naive bayes
gnb = GaussianNB()
y_pred = gnb.fit(dataset,target).predict(dataset)
print("GaussianNB mislabeled points out of a total %d points : %d" % (dataset.shape[0], (target != y_pred).sum()))
print gnb.score(dataset,y_pred)

bnb = BernoulliNB()
y_pred = gnb.fit(dataset,target).predict(dataset)
print("BernoulliNB mislabeled points out of a total %d points : %d" % (dataset.shape[0], (target != y_pred).sum()))
print bnb.score(dataset,y_pred)



## PREDICT ON TEST SET
# testset = read_csv('data/test.csv', header=0)
# train = dataset[dataset.columns[2:]]
# testset = testset[train.columns]

