import numpy
import numpy as np
import pandas as pd
from keras.losses import MeanSquaredLogarithmicError
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizer_v1 import SGD
from keras.wrappers.scikit_learn import KerasRegressor
from matplotlib import pyplot as plt, pyplot
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from random_forest_bagging import adding_column_class, split_dataset


def ann_algorithm():
    print('-------------------ANN------------------')
    dataset = pd.read_csv('data-set/result.csv')
    # dataset = adding_column_class(dataset)
    print(dataset)
    X = pd.DataFrame(dataset.drop(['Country', 'Rating'], axis=1)).values
    y = dataset['Rating'].values

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    # y_train = np.array(y_train).reshape(-1, 1)
    X_test = sc.transform(X_test)
    # y_test = np.array(y_test).reshape(-1, 1)
    # print(y_train.shape)
    # print(X_train)
    # print(y_test.shape)

    classifier = Sequential()

    classifier.add(Dense(units=30, activation='relu', input_dim=21))
    classifier.add(Dense(units=15, activation='relu'))
    classifier.add(Dense(units=1, activation='relu'))

    classifier.compile(optimizer='adam', loss='mse')
    history = classifier.fit(X_train, y_train, batch_size=15, validation_data=(X_test, y_test), epochs=50, verbose=0)
    # evaluate the model
    train_mse = classifier.evaluate(X_train, y_train, verbose=0)
    test_mse = classifier.evaluate(X_test, y_test, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_mse, test_mse))
    # plot loss during training
    pyplot.title('Loss / Mean Squared Error')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()
    #
    # errors = abs(y_pred[:, 1] - y_test)
    # mape = 100 * (errors / y_test)
    # # Calculate and display accuracy
    # accuracy = 100 - np.mean(mape)
    # print('Accuracy:', round(accuracy, 2), '%.')


if __name__ == '__main__':
    ann_algorithm()
