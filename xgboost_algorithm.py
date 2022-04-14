import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import absolute
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score, KFold
from xgboost import XGBClassifier, XGBRegressor

import build_dataset
import random_forest_bagging


def xgboost():
    print("-------------------XGBoost------------------------")
    # dataset, countries = build_dataset.build_imputed_dataset()
    dataset = pd.read_csv('data-set/result.csv')
    dataset = random_forest_bagging.adding_column_class(dataset)
    # print(df1.columns)
    X = dataset[[i for i in dataset.columns.tolist() if i != 'Overall rank' and i != 'Country' and i != 'Rating']]
    y = dataset['Rating']
    X_train, X_test, y_train, y_test = random_forest_bagging.split_dataset(X, y)
    # fit model no training data
    model = XGBRegressor(use_label_encoder=False, random_state=100)
    eval_set = [(X_test, y_test)]
    model.fit(X_train, y_train)
    score = model.score(X_train, y_train)
    print("Training score: ", score)
    # print(model.score(X_test, y_test))
    # print()

    # # - cross validataion
    # kfold = KFold(n_splits=10, shuffle=True)
    # kf_cv_scores = cross_val_score(model, X_train, y_train, cv=kfold)
    # print("K-fold CV average score: %.2f" % kf_cv_scores.mean())
    print('10-Fold Cross validation:', round(np.mean(cross_val_score(model, X_train, y_train, cv=10)), 2))

    ypred = model.predict(X_test)

    print('Mean Absolute Error:', round(mean_absolute_error(y_test, ypred), 2))
    print('Mean Squared Error:', round(mean_squared_error(y_test, ypred), 2))
    print('Root Mean Squared Error:', round(np.sqrt(mean_squared_error(y_test, ypred)), 2))

    x_ax = range(len(y_test))
    plt.plot(x_ax, y_test, label="original")
    plt.plot(x_ax, ypred, label="predicted")
    plt.title("Test and predicted data")
    plt.legend()
    plt.show()
