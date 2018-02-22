import pandas as pd
import logging
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint

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

    # one-hot encode categorical variables
    enc = OneHotEncoder(sparse=False)
    enc.fit(trainset)
    trainset = enc.transform(trainset)

    X_train, X_test, Y_train, Y_test = train_test_split(trainset, y_truth, test_size=0.1)

    nn = Sequential()
    nn.add(Dense(1000, input_dim=trainset.shape[1], activation='relu'))
    nn.add(Dropout(0.2))
    nn.add(Dense(300,  activation='relu'))
    nn.add(Dropout(0.2))
    nn.add(Dense(50,  activation='relu'))

    nn.add(Dense(1, activation='sigmoid'))  # output layer
    nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    filepath = "nn_models/nn_weights-{epoch:02d}.hdf5"
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=0,
                                                 save_weights_only=False, save_best_only=False, mode='max')

    fit_model = nn.fit(X_train, Y_train, epochs=100, verbose=1, batch_size=X_train.shape[0],
                        initial_epoch=0) #, callbacks=[checkpoint]

    nn.evaluate(X_test, Y_test, verbose=0)

    # Top up - Train on all the data for the competition:
    fit_model = nn.fit(X_test, Y_test, epochs=10, verbose=1, batch_size=X_test.shape[0],
                       initial_epoch=0)

    # get testing data
    testset = pd.read_csv('data/test.csv', header=0)
    testset = testset[cols2keep]
    testset = enc.transform(testset)

    testset = np.round(nn.predict(testset))
    # write to file
    y_out = pd.DataFrame(testset, columns=['is_female'])
    y_out.to_csv('results_logistic_regression.csv', index_label='test_id')



if __name__ == "__main__":
    start_time = time.time()
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s    %(levelname)s:%(message)s')
    main()
    logging.info("process ran for " + str(time.time() - start_time) + " seconds")