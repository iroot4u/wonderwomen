import pandas as pd
import logging
import time
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier


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

    # create train and test set for each algorithm
    itrain = range(int(0.7 * X_train.shape[0]))
    itest = range(itrain[len(itrain)-1], X_train.shape[0])


    # Try a whole bunch of different models
    # see: http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
    #-----------------------------------------
    models = [MLPClassifier(),
              LogisticRegression(),
              BernoulliNB(),
              RandomForestClassifier(),
              AdaBoostClassifier(),
              KNeighborsClassifier(),
              SVC(kernel='linear'),
              SVC()
              ]

    modelNames = ['MLP Classifier',
                  'Logistic Regression',
                  'Bernoulli NB',
                  'Random Forest',
                  'Ada Boost',
                  'K Nearest Neighbors',
                  'Linear SVM',
                  'RBF SVM'
                  ]

    for model, name in zip(models, modelNames):
        model.fit(X_train[itrain], y_truth[itrain])
        logging.info(name + ': ' + str(model.score(X_train[itest], y_truth[itest])))




    # # Make predictions and write to file for competition
    # # train logistic regression
    # model = LogisticRegression()
    # model.fit(X_train, y_truth)
    #
    # # get testing data
    # testset = pd.read_csv('data/test.csv', header=0)
    # testset = testset[cols2keep]
    # X_test = enc.transform(testset)
    #
    # # predict on testset
    # y_predict = model.predict(X_test)
    #
    # # write to file
    # y_out = pd.DataFrame(y_predict, columns=['is_female'])
    # y_out.to_csv('results_logistic_regression.csv', index_label='test_id')



if __name__ == "__main__":
    start_time = time.time()
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s  %(levelname)s:  %(message)s')
    main()
    logging.info("process ran for " + str(time.time() - start_time) + " seconds")