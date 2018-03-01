print(__doc__)
from pandas import read_csv
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from numpy import genfromtxt, savetxt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing
from sklearn.cross_validation import cross_val_score

features2keep = read_csv('most_important_features.csv')
features2keep = features2keep['feature'].values.tolist()
features2keep.append('is_female')

dataset = read_csv('data/train.csv', header=0)
target = dataset['is_female']
dataset.drop(labels=['is_female'], axis=1, inplace=True)
dataset.insert(0, 'is_female', target)
random = pd.DataFrame()
# random['randNumCol'] = np.random.randint(1, 6, dataset.shape[0])
# dataset.insert(len(dataset.columns), 'randNumCol', random)
# print len(dataset.columns)
dataset = dataset.dropna(axis=1)
#dataset = dataset.fillna(-1)
# print len(dataset.columns)
dataset = dataset.filter(items=features2keep)

X_train = dataset.drop(labels=['is_female'], axis=1)
y_truth = dataset['is_female']


# cross-validate random forest for error estimation
itrain = range(int(0.7*len(dataset)))
itest = range(itrain[len(itrain)-1]+1, len(dataset))
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train.iloc[itrain], y_truth[itrain])
print('random forest accuracy = ' + str(rf.score(X_train.iloc[itest], y_truth[itest])))



# read in test set and apply random forest
testset = read_csv('data/test.csv', header=0)
testset = testset.filter(items=features2keep)
# features = list(train.columns.values)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_truth)

# predict on testset
y_predict = rf.predict(testset)

# write to file
y_out = pd.DataFrame(y_predict, columns=['is_female'])
y_out.to_csv('results_rf_features_subset.csv', index_label='test_id')



# importances = rf.feature_importances_
# std = np.std([tree.feature_importances_ for tree in rf.estimators_],
#              axis=0)
# indices = np.argsort(importances)[::-1]
#
# # print the feature ranking
# print("Feature ranking:")
#
# for f in range(train.shape[1]):
#     print("%d. feature %d %s (%f)" % (f + 1, indices[f], features[indices[f]], importances[indices[f]]))
#
# # plot the feature importances of the forest
# plt.figure()
# plt.title("Feature importances")
# plt.bar(range(train.shape[1]), importances[indices],
#        color="r", yerr=std[indices], align="center")
# plt.xticks(range(train.shape[1]), indices)
# plt.xlim([-1, train.shape[1]])
# plt.show()

# testset = testset[train.columns]  # don't need this for random feature test
# y_predict = rf.predict(testset)
# y_out = pd.DataFrame(y_predict, columns=['is_female'])
# y_out.to_csv('results_rf.csv', index_label='test_id')
#

#**** old code*******
# x_train, x_test, y_train, y_test = train_test_split(train, target, test_size=0.4, random_state=0)
#
# print x_train.shape, y_train.shape
# print x_train.shape, y_train.shape
#
# dataset.fillna(-1)
#
# coltypes = dataset.columns.to_series().groupby(dataset.dtypes).groups
# dict = {k.name: v for k, v in coltypes.items()}
#
# number = preprocessing.LabelEncoder()
# for obj in dict['object']:
#     #print obj
#     #print dataset[str(obj)].dtype
#     dataset[str(obj)] = number.fit_transform(dataset[str(obj)])
#     #print dataset[str(obj)].dtype
#
#
# dataset.fillna(-1)
# dataset.isnull().any()
#
# print dataset['LN2_RIndLngBEOth']
# train = dataset[0:]
#
# x_train, x_test, y_train, y_test = train_test_split(train, target, test_size=0.4, random_state=0)
# #print x_train.shape, y_train.shape
# #print x_train.shape, y_train.shape
#
# rf = RandomForestClassifier(n_estimators=100)
# print np.mean(cross_val_score(rf, x_train, y_train, cv=10))
#
#
# #rf.fit(train, target)
# # def main():
# #     # create the training & test sets, skipping the header row with [1:]
# #     dataset = genfromtxt(open('data/train.csv', 'r'), delimiter=',', dtype='f8',)[1:]
# #     target = [x[0] for x in dataset]
# #     train = [x[1:] for x in dataset]
# #     test = genfromtxt(open('data/test.csv', 'r'), delimiter=',', dtype='f8')[1:]
# #
# #     # create and train the random forest
# #     # multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
# #     rf = RandomForestClassifier(n_estimators=100)
# #     rf.fit(train, target)
# #
# #     savetxt('Data/submission2.csv', rf.predict(test), delimiter=',', fmt='%f')
# #
# #
# if __name__ == "__main__":
#     main()