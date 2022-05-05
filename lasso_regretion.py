# use automatically configured the lasso regression algorithm
import warnings
# load the dataset
import numpy
import shap
from numpy import arange
from sklearn.linear_model import LassoCV
from sklearn.model_selection import RepeatedKFold, cross_val_score
import pandas as pd
from sklearn.preprocessing import StandardScaler

import random_forest_bagging
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


def lasso():
    warnings.simplefilter("ignore")
    dataset = pd.read_csv('data-set/result.csv')
    # dataset = random_forest_bagging.adding_column_class(dataset)
    # print(df1.columns)
    X = dataset[[i for i in dataset.columns.tolist() if i != 'Overall rank' and i != 'Country' and i != 'Rating' and i != 'Life satisfaction']]
    y = dataset['Rating']
    # define model evaluation method
    lasso_regretion(X, y)


def lasso_regretion(X, y):
    print('------------LASSO REGRETION-------------')
    cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)
    # define model
    model = LassoCV(alphas=arange(0, 1, 0.01), cv=cv, n_jobs=-1)

    X_train, X_test, y_train, y_test = split_dataset(X, y)
    scaler = StandardScaler().fit(X_train[X.columns])

    X_train[X.columns] = scaler.transform(X_train[X.columns])

    X_test[X.columns] = scaler.transform(X_test[X.columns])
    model.fit(X_train, y_train)

    # print('Score:', model.score(X,y))
    y_pred = model.predict(X_test)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', numpy.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('R2 square:', round(metrics.r2_score(y_test, y_pred), 2))
    score = cross_val_score(model, X_train, y_train, cv=10)
    print("Cross Validation Scores are {}".format(score))
    print("Average Cross Validation score :{}".format(score.mean()))

    errors = abs(y_pred - y_test)
    mape = 100 * (errors / y_test)
    # Calculate and display accuracy
    accuracy = 100 - numpy.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')
    print(model.score(X_test, y_test))

    explainer = shap.Explainer(model.predict, X_train)
    # Calculates the SHAP values - It takes some time
    shap_values = explainer(X_test)
    # Evaluate SHAP values
    shap.plots.bar(shap_values)

    x_ax = range(len(y_test))
    plt.plot(x_ax, y_test, label="original")
    plt.plot(x_ax, y_pred, label="predicted")
    plt.title("LASSO REGRETION-Test and predicted data")
    plt.legend()
    plt.show()

    return y_pred


def split_dataset(X, y):
    return train_test_split(X, y, test_size=0.25, random_state=100)
