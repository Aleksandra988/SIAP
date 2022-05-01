import numpy
import numpy as np
import pandas as pd
from keras.losses import MeanSquaredLogarithmicError
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from matplotlib import pyplot as plt
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

    classifier.add(Dense(units=10, activation='relu', input_dim=21, kernel_initializer='normal'))
    classifier.add(Dense(units=10, activation='relu', kernel_initializer='normal'))
    classifier.add(Dense(units=1, kernel_initializer='normal'))

    # loss function
    msle = MeanSquaredLogarithmicError()
    classifier.compile(optimizer='adam', loss=msle, metrics=['mse', 'mae'])
    classifier.fit(X_train, y_train, batch_size=20, epochs=50, verbose=1)

    y_pred = classifier.predict(X_test)
    error = mean_absolute_error(y_test, y_pred)
    print('MAE: %.3f' % error)
    #
    # errors = abs(y_pred[:, 1] - y_test)
    # mape = 100 * (errors / y_test)
    # # Calculate and display accuracy
    # accuracy = 100 - np.mean(mape)
    # print('Accuracy:', round(accuracy, 2), '%.')


if __name__ == '__main__':
    ann_algorithm()
