import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier


def rf():
    print('------------RANDOM FOREST CLASSIFICATION-------------')
    dataset = pd.read_csv('data-set/result.csv')
    dataset = adding_column_class(dataset)

    X = dataset.drop(['Rating', 'Country', 'class', 'Rating'], axis=1)
    y = dataset['class']
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    clf = RandomForestClassifier(random_state=100)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print(clf.score(X_train, y_train))
    # print(clf.score(X_test, y_test))
    print("Accuracy:", round(metrics.accuracy_score(y_test, preds), 2))
    # 10-Fold Cross validation
    # print(np.mean(cross_val_score(clf, X_train, y_train, cv=10)))
    print(classification_report(y_test, preds, zero_division=0))


def rf_regression():
    print('------------RANDOM FOREST REGRESSION-------------')
    dataset = pd.read_csv('data-set/result.csv')
    X = dataset.drop(['Rating', 'Country'], axis=1)
    y = dataset['Rating']
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators=1000, random_state=100)
    rf.fit(X_train, y_train);
    y_pred = rf.predict(X_test)
    print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test, y_pred), 2))
    print('Mean Squared Error:', round(metrics.mean_squared_error(y_test, y_pred), 2))
    print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), 2))

    # # Calculate the absolute errors
    errors = abs(y_pred - y_test)
    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / y_test)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')
    # print(rf.score(X_test, y_test))
    # 10-Fold Cross validation
    print('10-Fold Cross validation:', round(np.mean(cross_val_score(rf, X_train, y_train, cv=10)), 2))

    x_ax = range(len(y_test))
    plt.plot(x_ax, y_test, label="original")
    plt.plot(x_ax, y_pred, label="predicted")
    plt.title("Boston test and predicted data")
    plt.legend()
    plt.show()


def adding_column_class(df3):
    df3['class'] = 0
    df3['class'] = np.where(df3['Rating'] < 6, 1, df3['class'])
    df3['class'] = np.where((df3['Rating'] >= 6) & (df3['Rating'] <= 7), 2, df3['class'])
    df3['class'] = np.where(df3['Rating'] > 7, 3, df3['class'])
    return df3


def split_dataset(X, y):
    return train_test_split(X, y, test_size=0.25, random_state=100)


def bagging():

    print('------------BAGGING-------------')
    dataset = pd.read_csv('data-set/result.csv')
    dataset = adding_column_class(dataset)

    X = dataset.drop(['Rating', 'Country', 'class', 'Rating'], axis=1)
    y = dataset['class']

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    print('Bagging with random forest classifier:')
    bag_model_rf = BaggingClassifier(RandomForestClassifier(), random_state=100)
    bag_model_rf.fit(X_train, y_train)
    print(round(bag_model_rf.score(X_train, y_train), 2))
    preds = bag_model_rf.predict(X_test)
    print("Accuracy:", round(metrics.accuracy_score(y_test, preds), 2))

    print('Bagging with KN classifier:')
    bag_model_kn = BaggingClassifier(KNeighborsClassifier(), random_state=100)
    bag_model_kn.fit(X_train, y_train)
    # print(bag_model.oob_score_)
    print(round(bag_model_kn.score(X_train, y_train), 2))
    preds = bag_model_kn.predict(X_test)
    print("Accuracy:", round(metrics.accuracy_score(y_test, preds), 2))
