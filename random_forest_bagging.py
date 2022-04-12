import math

import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
# from build_dataset import build_imputed_dataset
import build_dataset
import random
random.seed(0)


def rf():
    # dataset, countries = build_dataset.build_imputed_dataset()
    dataset = pd.read_csv('data-set/result.csv')
    # print(dataset)
    print('------------RANDOM FOREST-------------')
    # 2018.csv dataset
    dataset = adding_column_class(dataset)

    X = dataset.drop(['Rating', 'Country', 'class', 'Rating'], axis=1)
    y = dataset['class']
    # rf_training_and_prediction(X, y, 'NOVO')
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    # Normalize the data
    # sc = StandardScaler()
    # normed_train_data = pd.DataFrame(sc.fit_transform(training), columns=X.columns)
    # normed_test_data = pd.DataFrame(sc.fit_transform(testing), columns=X.columns)
    clf = RandomForestClassifier(random_state=100)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print(clf.score(X_train, y_train))
    print(clf.score(X_test, y_test))
    print("Accuracy:", metrics.accuracy_score(y_test, preds))
    # 10-Fold Cross validation
    # print(np.mean(cross_val_score(clf, X_train, y_train, cv=10)))


def rf_regression():
    # dataset, countries = build_dataset.build_imputed_dataset()
    print('------------RANDOM FOREST REGRESSION-------------')
    dataset = pd.read_csv('data-set/result.csv')
    X = dataset.drop(['Rating', 'Country'], axis=1)
    y = dataset['Rating']
    # plt.hist(dataset['Rating'])
    # plt.xlabel('Rating')
    # plt.ylabel('Frequency')
    # plt.show()
    # Labels are the values we want to predict
    labels = np.array(dataset['Rating'])
    # Remove the labels from the features
    # axis 1 refers to the columns
    features = dataset.drop('Rating', axis=1)
    features = features.drop('Country', axis=1)
    # Saving feature names for later use
    feature_list = list(features.columns)
    # Convert to numpy array
    features = np.array(features)
    # rf_training_and_prediction_regression(X4, y4, "NOVO")
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    # print('Training Features Shape:', X_train.shape)
    # print('Training Labels Shape:', y_train.shape)
    # print('Testing Features Shape:', X_test.shape)
    # print('Testing Labels Shape:', y_test.shape)

    # baseline_preds = test_features[:, feature_list.index('average')]
    # # Baseline errors, and display average baseline error
    # baseline_errors = abs(baseline_preds - test_labels)
    # print('Average baseline error: ', round(np.mean(baseline_errors), 2))

    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators=1000, random_state=100)
    # Train the model on training data
    rf.fit(X_train, y_train);

    # Use the forest's predict method on the test data
    y_pred = rf.predict(X_test)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    # # Calculate the absolute errors
    errors = abs(y_pred - y_test)
    # # # Print out the mean absolute error (mae)
    # print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / y_test)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')
    print(rf.score(X_test, y_test))
    # print(rf.score(X_test, y_test))
    # 10-Fold Cross validation
    # print(np.mean(cross_val_score(rf, X_train, y_train, cv=10)))


def adding_column_class(df3):
    df3['class'] = 0
    df3['class'] = np.where(df3['Rating'] < 4, 1, df3['class'])
    # df3['rating'] = np.where((df3['Score'] >= 3) & (df3['Score'] <= 4), 2, df3['rating'])
    df3['class'] = np.where((df3['Rating'] >= 4) & (df3['Rating'] <= 5), 2, df3['class'])
    df3['class'] = np.where((df3['Rating'] >= 5) & (df3['Rating'] <= 6), 3, df3['class'])
    df3['class'] = np.where((df3['Rating'] >= 6) & (df3['Rating'] <= 7), 4, df3['class'])
    df3['class'] = np.where(df3['Rating'] > 7, 5, df3['class'])
    return df3


def split_dataset(X, y):
    return train_test_split(X, y, test_size=0.25, random_state=100)


def bagging():

    print('------------BAGGING-------------')
    # dataset, countries = build_dataset.build_imputed_dataset()
    dataset = pd.read_csv('data-set/result.csv')
    dataset = adding_column_class(dataset)

    # 2018.csv dataset
    X = dataset[[i for i in dataset.columns.tolist() if i != 'Overall rank' and i != 'Country' and i != 'Rating' and i != 'class']]
    y = dataset['class']

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    print('Bagging with random forest classifier:')
    bag_model_rf = BaggingClassifier(RandomForestClassifier(random_state=100), random_state=100)
    bag_model_rf.fit(X_train, y_train)
    # print(bag_model.oob_score_)
    print(bag_model_rf.score(X_train, y_train))
    preds = bag_model_rf.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, preds))

    print('Bagging with KN classifier:')
    bag_model_kn = BaggingClassifier(KNeighborsClassifier(), random_state=100)
    bag_model_kn.fit(X_train, y_train)
    # print(bag_model.oob_score_)
    print(bag_model_kn.score(X_train, y_train))
    preds = bag_model_kn.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, preds))

    print()
    # scores = cross_val_score(RandomForestClassifier(n_estimators=50), X, y)
    # scores.mean()
