import pandas as pd
import logging
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
import numpy as np



def main():
    # get and clean training data
    trainset = pd.read_csv('data/train.csv', header=0)
    y_truth = trainset['is_female']  # store labels in a new variable
    trainset.drop(labels=['train_id', 'is_female'], axis=1, inplace=True)  # remove is_female and train_id from training set

    # drop columns with NA and columns with more than 20 unique values
    # logging.info('total number of features: ' + str(len(dataset.columns)))
    trainset = trainset.dropna(axis=1)  # remove features that contain NA values
    toDrop = trainset.apply(lambda x: len(np.unique(x)) > 20, axis='rows')
    trainset = trainset.drop(labels=trainset.columns[toDrop], axis=1)
    # logging.info('number of features after dropping NA: ' + str(len(dataset.columns)))
    cols2keep = trainset.columns

    # # get dtype of each column
    # for col in trainset.columns:
    #     logging.info('\t column name: ' + col + ':' + str(trainset[col].dtype))

    # one-hot encode categorical variables
    enc = OneHotEncoder()
    enc.fit(trainset)
    X_train = enc.transform(trainset)

    # train logistic regression with cross-validation for error checking
    logging.info('cross-validating logistic regression...')
    modelcv = LogisticRegressionCV(cv=3)
    modelcv.fit(X_train, y_truth)
    logging.info('cross-validation score is '+ str(modelcv.scores_))


    # train logistic regression
    model = LogisticRegression()
    model.fit(X_train, y_truth)

    # get testing data
    testset = pd.read_csv('data/test.csv', header=0)
    testset = testset[cols2keep]
    X_test = enc.transform(testset)

    # predict on testset
    y_predict = model.predict(X_test)

    # write to file
    y_out = pd.DataFrame(y_predict, columns=['is_female'])
    y_out.to_csv('results_logistic_regression.csv', index_label='test_id')



if __name__ == "__main__":
    start_time = time.time()
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s    %(levelname)s:%(message)s')
    main()
    logging.info("process ran for " + str(time.time() - start_time) + " seconds")