import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from scipy.special import softmax


def rf():
    print('------------RANDOM FOREST CLASSIFICATION-------------')
    dataset = pd.read_csv('data-set/result.csv')
    dataset = adding_column_class(dataset)

    X = dataset.drop(['Rating', 'Country', 'class', 'Rating'], axis=1)
    y = dataset['class']
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    clf = RandomForestClassifier(random_state=35)
    clf.fit(X_train, y_train)
    # print(clf.feature_importances_)
    # plt.barh(X_test.columns, clf.feature_importances_)
    # plt.show()
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

    # # scaling features
    # scalar = StandardScaler()
    #
    # # fit and transform scalar to train set
    # X_trainn = scalar.fit_transform(X_train)
    #
    # # transform test set
    # X_test = scalar.transform(X_test)

    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators=1000, random_state=35)
    rf.fit(X_train, y_train)
    # print(rf.feature_importances_)
    # plt.barh(X_train.columns, rf.feature_importances_)
    # plt.show()
    print(dataset['Life satisfaction'])
    y_pred = rf.predict(X_test)
    print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test, y_pred), 2))
    print('Mean Squared Error:', round(metrics.mean_squared_error(y_test, y_pred), 2))
    print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), 2))
    print('R2 square:', round(metrics.r2_score(y_test, y_pred), 2))

    # 10-Fold Cross validation
    score = cross_val_score(rf, X, y, cv=10)
    print("Cross Validation Scores are {}".format(score))
    print("Average Cross Validation score :{}".format(score.mean()))

    # # # Calculate the absolute errors
    # errors = abs(y_pred - y_test)
    # # Calculate mean absolute percentage error (MAPE)
    # mape = 100 * (errors / y_test)
    # # Calculate and display accuracy
    # accuracy = 100 - np.mean(mape)
    # print('Accuracy:', round(accuracy, 2), '%.')
    # print(rf.score(X_test, y_test))

    # Fits the explainer
    explainer = shap.Explainer(rf.predict, X_train)
    # Calculates the SHAP values - It takes some time
    shap_values = explainer(X_test)
    # Evaluate SHAP values
    shap.plots.bar(shap_values)

    # x_ax = range(len(y_test))
    # plt.plot(x_ax, y_test, label="original")
    # plt.plot(x_ax, y_pred, label="predicted")
    # plt.title("Test and predicted data")
    # plt.legend()
    # plt.show()


def adding_column_class(df3):
    df3['class'] = 0
    df3['class'] = np.where(df3['Rating'] < 6, 1, df3['class'])
    df3['class'] = np.where((df3['Rating'] >= 6) & (df3['Rating'] <= 7), 2, df3['class'])
    df3['class'] = np.where(df3['Rating'] > 7, 3, df3['class'])
    return df3


def split_dataset(X, y):
    return train_test_split(X, y, test_size=0.25, random_state=35)


def bagging():

    print('------------BAGGING-------------')
    dataset = pd.read_csv('data-set/result.csv')
    dataset = adding_column_class(dataset)

    X = dataset.drop(['Rating', 'Country', 'class', 'Rating'], axis=1)
    y = dataset['class']

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    print('Bagging with random forest classifier:')
    bag_model_rf = BaggingClassifier(RandomForestClassifier(), random_state=35)
    bag_model_rf.fit(X_train, y_train)
    print(round(bag_model_rf.score(X_train, y_train), 2))
    preds = bag_model_rf.predict(X_test)
    print("Accuracy:", round(metrics.accuracy_score(y_test, preds), 2))
    print(classification_report(y_test, preds, zero_division=0))

    print('Bagging with KN classifier:')
    bag_model_kn = BaggingClassifier(KNeighborsClassifier(), random_state=35)
    bag_model_kn.fit(X_train, y_train)
    # print(bag_model.oob_score_)
    print(round(bag_model_kn.score(X_train, y_train), 2))
    preds = bag_model_kn.predict(X_test)
    print("Accuracy:", round(metrics.accuracy_score(y_test, preds), 2))


def print_feature_importances_shap_values(shap_values, features):
    '''
    Prints the feature importances based on SHAP values in an ordered way
    shap_values -> The SHAP values calculated from a shap.Explainer object
    features -> The name of the features, on the order presented to the explainer
    '''
    # Calculates the feature importance (mean absolute shap value) for each feature
    importances = []
    for i in range(shap_values.values.shape[1]):
        importances.append(np.mean(np.abs(shap_values.values[:, i])))
    # Calculates the normalized version
    importances_norm = softmax(importances)
    # Organize the importances and columns in a dictionary
    feature_importances = {fea: imp for imp, fea in zip(importances, features)}
    feature_importances_norm = {fea: imp for imp, fea in zip(importances_norm, features)}
    # Sorts the dictionary
    feature_importances = {k: v for k, v in sorted(feature_importances.items(), key=lambda item: item[1], reverse = True)}
    feature_importances_norm= {k: v for k, v in sorted(feature_importances_norm.items(), key=lambda item: item[1], reverse = True)}
    # Prints the feature importances
    for k, v in feature_importances.items():
        print(f"{k} -> {v:.4f} (softmax = {feature_importances_norm[k]:.4f})")