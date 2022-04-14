import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.losses import MeanSquaredLogarithmicError
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from random_forest_bagging import adding_column_class


def ann_algorithm():
    print('-------------------ANN------------------')
    dataset = pd.read_csv('data-set/result.csv')
    # dataset = adding_column_class(dataset)
    print(dataset)
    X = pd.DataFrame(dataset.drop(['Country', 'Rating'], axis=1)).values
    y = dataset['Rating'].values
    # X = pd.DataFrame(dataset.iloc[:, [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23]].values)
    # y = dataset.iloc[:, 15].values
    print(y)

    # labelencoder_X_1 = LabelEncoder()
    # X.loc[:, 1] = labelencoder_X_1.fit_transform(X.iloc[:, 2])
    #
    # ct = ColumnTransformer([("Geography", OneHotEncoder(), [1])], remainder='passthrough')
    # X = ct.fit_transform(X)
    # labelencoder_X2 = LabelEncoder()
    # X[:, 2] = labelencoder_X2.fit_transform(X[:, 2])
    # X = X[:, 1:]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    # y_train = np.array(y_train).reshape(-1, 1)
    X_test = sc.transform(X_test)
    # y_test = np.array(y_test).reshape(-1, 1)
    # print(y_train.shape)
    # print(X_train)
    print(y_test.shape)

    classifier = Sequential()

    classifier.add(Dense(units=10, activation='relu', input_dim=21, kernel_initializer='normal'))
    classifier.add(Dense(units=10, activation='relu', kernel_initializer='normal'))
    classifier.add(Dense(units=10, kernel_initializer='normal'))

    # loss function
    msle = MeanSquaredLogarithmicError()
    # model.compile(
    #     loss=msle,
    #     optimizer=Adam(learning_rate=learning_rate),
    #     metrics=[msle]
    # )
    # classifier.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    classifier.compile(optimizer='adam', loss=msle, metrics=['mse', 'mae'])
    classifier.fit(X_train, y_train, batch_size=100, epochs=200)

    y_pred = classifier.predict(X_test)
    # y_pred = (y_pred > 0.5)
    # print(y_pred)

    # cm = confusion_matrix(y_test, y_pred)
    # print(cm)
    # accuracy_score(y_test, y_pred)
    # score = classifier.evaluate(X_test, y_test, verbose=0)
    # print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
    # evaluate model

    # Computing the absolute percent error
    # APE = 100 * (abs(TestingData['Price'] - TestingData['PredictedPrice']) / TestingData['Price'])
    # TestingData['APE'] = APE
    #
    # print('The Accuracy of ANN model is:', 100 - np.mean(APE))
    #
    # estimator = KerasRegressor(build_fn=classifier, epochs=100, batch_size=5, verbose=0)
    # kfold = KFold(n_splits=10)
    # results = cross_val_score(estimator, X, y, cv=kfold)
    # print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))

    # onehotencoder = OneHotEncoder(categorical_features=[2])
    # labelencoder_X_1 = LabelEncoder()
    # X.loc[:, 2] = labelencoder_X_1.fit_transform(X.iloc[:, 2])
    # X = onehotencoder.fit_transform(X).toarray()
    # X = X[:, 2:]

    # print(dataset.iloc[:, 16])

    # X = dataset.drop(['Rating', 'Country', 'Rating'], axis=1)
    # y = dataset['Rating']
    # print(y)
    # y = np.array(y).reshape(-1, 1)
    # print(y)
    #
    # ### Sandardization of data ###
    # from sklearn.preprocessing import StandardScaler
    # PredictorScaler = StandardScaler()
    # TargetVarScaler = StandardScaler()
    #
    # # Storing the fit object for later reference
    # PredictorScalerFit = PredictorScaler.fit(X)
    # TargetVarScalerFit = TargetVarScaler.fit(y)
    #
    # # Generating the standardized values of X and y
    # X = PredictorScalerFit.transform(X)
    # y = TargetVarScalerFit.transform(y)
    #
    # # Split the data into training and testing set
    # from sklearn.model_selection import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    #
    # # Quick sanity check with the shapes of Training and testing datasets
    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_test.shape)
    # print(y_test.shape)
    #
    # # create ANN model
    # model = Sequential()
    #
    # # Defining the Input layer and FIRST hidden layer, both are same!
    # model.add(Dense(units=5, input_dim=5, kernel_initializer='normal', activation='relu'))
    #
    # # Defining the Second layer of the model
    # # after the first layer we don't have to specify input_dim as keras configure it automatically
    # model.add(Dense(units=5, kernel_initializer='normal', activation='tanh'))
    #
    # # The output neuron is a single fully connected node
    # # Since we will be predicting a single number
    # model.add(Dense(1, kernel_initializer='normal'))
    #
    # # Compiling the model
    # model.compile(loss='mean_squared_error', optimizer='adam')
    #
    # # Fitting the ANN to the Training set
    # model.fit(X_train, y_train, batch_size=20, epochs=50, verbose=1)


if __name__ == '__main__':
    ann_algorithm()
