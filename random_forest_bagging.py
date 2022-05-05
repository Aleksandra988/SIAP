import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, RandomForestRegressor, BaggingRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from scipy.special import softmax
from sklearn.model_selection import KFold
import seaborn as sns


def rf_regression():
    print('------------RANDOM FOREST REGRESSION-------------')
    dataset = pd.read_csv('data-set/result.csv')
    X = dataset.drop(['Rating', 'Country', 'Life satisfaction'], axis=1)
    y = dataset['Rating']
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    rf = RandomForestRegressor(random_state=35)
    rf.fit(X_train, y_train)
    # print("Feature importance: ")
    # print(rf.feature_importances_)
    # plt.barh(X_train.columns, rf.feature_importances_)
    # plt.show()
    y_pred = rf.predict(X_test)
    print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test, y_pred), 2))
    print('Mean Squared Error:', round(metrics.mean_squared_error(y_test, y_pred), 2))
    print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), 2))
    print('R2 square:', round(metrics.r2_score(y_test, y_pred), 2))

    # 10-Fold Cross validation
    score = cross_val_score(rf, X_train, y_train, cv=10)
    print("Cross Validation Scores are {}".format(score))
    print("Average Cross Validation score :{}".format(score.mean()))

    # Grid Search for best hyperparameters
    hyperparameter_space = {
        # 'bootstrap': [True],
        'max_depth': [2, 3, 4, 6, 8, 10, 12, 15, 20],
        # 'max_features': [2, 3],
        'min_samples_leaf': [1, 2, 4, 6, 8, 10, 15, 20, 30],
        'min_samples_split': [2, 3, 4, 5, 6, 8, 10, 12],
        'n_estimators': [100, 120, 200]
    }

    gs = GridSearchCV(rf, param_grid=hyperparameter_space, scoring="r2", n_jobs=-1, cv=10,
                      return_train_score=True)

    gs.fit(X_train, y_train)
    print("Optimal hyperparameter combination:", gs.best_params_)
    print()
    print("Mean cross-validated training accuracy score:", round(gs.best_score_, 2))

    y_predd = gs.predict(X_test)
    print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test, y_predd), 2))
    print('Mean Squared Error:', round(metrics.mean_squared_error(y_test, y_predd), 2))
    print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test, y_predd)), 2))
    print('R2 square:', round(metrics.r2_score(y_test, y_predd), 2))

    # # # Calculate the absolute errors
    # errors = abs(y_pred - y_test)
    # # Calculate mean absolute percentage error (MAPE)
    # mape = 100 * (errors / y_test)
    # # Calculate and display accuracy
    # accuracy = 100 - np.mean(mape)
    # print('Accuracy:', round(accuracy, 2), '%.')
    # print(rf.score(X_test, y_test))

    # SHAP values
    # Fits the explainer
    explainer = shap.Explainer(rf.predict, X_train)
    # Calculates the SHAP values - It takes some time
    shap_values = explainer(X_train)
    # Evaluate SHAP values
    shap.plots.bar(shap_values)
    # shap.summary_plot(shap_values, X_train, plot_type='dot', plot_size=[50, 6])

    x_ax = range(len(y_test))
    plt.plot(x_ax, y_test, label="original")
    plt.plot(x_ax, y_pred, label="predicted")
    plt.title("RANDOM FOREST - Test and predicted data")
    plt.legend()
    plt.show()


def cross_validate(features, target, classifier, k_fold):
    '''Calculates average accuracy of classification
    algorithm using kfold crossvalidation'''
    # derive a set of (random) training and testing indices
    fold = KFold(n_splits=k_fold, shuffle=False)
    # for each training and testing slices run the classifier, and score the results
    k_score_total = 0
    for train_slice, test_slice in fold.split(features):
        model = classifier.fit(features[train_slice],
                               target[train_slice])
        k_score = model.score(features[test_slice],
                              target[test_slice])
        k_score_total += k_score
    # return the average accuracy
    return k_score_total / k_fold


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
    # dataset = adding_column_class(dataset)

    X = dataset.drop(['Rating', 'Country', 'Life satisfaction'], axis=1)
    y = dataset['Rating']

    X_train, X_test, y_train, y_test = split_dataset(X, y)
    rff = RandomForestRegressor(max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=200)

    print('Bagging with random forest classifier:')
    bag_model_rf = BaggingRegressor(rff, random_state=35, n_estimators=10)
    bag_model_rf.fit(X_train, y_train)
    print(round(bag_model_rf.score(X_train, y_train), 2))
    y_pred = bag_model_rf.predict(X_test)
    print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test, y_pred), 2))
    print('Mean Squared Error:', round(metrics.mean_squared_error(y_test, y_pred), 2))
    print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), 2))
    print('R2 square:', round(metrics.r2_score(y_test, y_pred), 2))

    # 10-Fold Cross validation
    score = cross_val_score(bag_model_rf, X_train, y_train, scoring='r2', cv=10, n_jobs=-1)
    print("Cross Validation Scores are {}".format(score))
    print("Average Cross Validation score :{}".format(score.mean()))
