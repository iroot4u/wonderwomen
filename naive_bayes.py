import logging
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from pandas import read_csv
from sklearn.preprocessing import OneHotEncoder

## import data
dataset = read_csv('data/train.csv', header=0)
target = dataset['is_female']

## remove training id and class
dataset.drop(labels=['train_id', 'is_female'], axis=1, inplace =True)
#dataset.insert(0, 'is_female', target)

## drop columns with NA
dataset = dataset.dropna(axis=1)
cols_no_na_mask = dataset.columns

## replace NA columns with value -1
# dataset = dataset.fillna(-1)

## one-hot encode categorical variables
enc = OneHotEncoder()
enc.fit(dataset)
sparsedataset = enc.transform(dataset)

## create train and test set for each algorithm
cutoff = int(0.7 * dataset.shape[0])
x_train = dataset[0:cutoff]
y_train = dataset[cutoff:dataset.shape[0]]
x_test = target[0:cutoff]
y_test = target[cutoff:]

## naive bayes classifiers
modelsNB = [GaussianNB(),
            BernoulliNB()
           ]

modelNamesNB = ['Gaussian Naive Bayes',
                'Bernoulli Naive Bayes'
               ]

for model, name in zip(modelsNB, modelNamesNB):
    model.fit(x_train, x_test)
    print(name + ': ' + str(model.score(y_train, y_test)))

