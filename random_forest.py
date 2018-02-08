from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from numpy import genfromtxt, savetxt
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import cross_val_score
import numpy as np

dataset = read_csv('data/train.csv', header=0)
target = dataset['is_female']
dataset.drop(labels=['is_female'], axis=1,inplace = True)
dataset.insert(0, 'is_female', target)

dataset.fillna(-1)

coltypes = dataset.columns.to_series().groupby(dataset.dtypes).groups
dict = {k.name: v for k, v in coltypes.items()}
number = preprocessing.LabelEncoder()
for obj in dict['object']:
    #print obj
    #print dataset[str(obj)].dtype
    dataset[str(obj)] = number.fit_transform(dataset[str(obj)])
    #print dataset[str(obj)].dtype

dataset.fillna(-1)
dataset.isnull().any()

train = dataset[0:]

x_train, x_test, y_train, y_test = train_test_split(train, target, test_size=0.4, random_state=0)
#print x_train.shape, y_train.shape
#print x_train.shape, y_train.shape

rf = RandomForestClassifier(n_estimators=100)
print np.mean(cross_val_score(rf, x_train, y_train, cv=10))

#rf.fit(train, target)


# def main():
#     # create the training & test sets, skipping the header row with [1:]
#     dataset = genfromtxt(open('data/train.csv', 'r'), delimiter=',', dtype='f8',)[1:]
#     target = [x[0] for x in dataset]
#     train = [x[1:] for x in dataset]
#     test = genfromtxt(open('data/test.csv', 'r'), delimiter=',', dtype='f8')[1:]
#
#     # create and train the random forest
#     # multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
#     rf = RandomForestClassifier(n_estimators=100)
#     rf.fit(train, target)
#
#     savetxt('Data/submission2.csv', rf.predict(test), delimiter=',', fmt='%f')
#
#
# if __name__ == "__main__":
#     main()