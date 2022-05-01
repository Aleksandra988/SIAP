import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from numpy import absolute
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score, KFold
from xgboost import XGBClassifier, XGBRegressor

import build_dataset
import random_forest_bagging


def xgboost():
    print("-------------------XGBoost------------------------")
    # dataset, countries = build_dataset.build_imputed_dataset()
    dataset = pd.read_csv('data-set/result.csv')
    # dataset = random_forest_bagging.adding_column_class(dataset)
    # print(df1.columns)
    X = dataset[[i for i in dataset.columns.tolist() if i != 'Overall rank' and i != 'Country' and i != 'Rating']]
    y = dataset['Rating']
    X_train, X_test, y_train, y_test = random_forest_bagging.split_dataset(X, y)
    # fit model no training data
    model = XGBRegressor(use_label_encoder=False, random_state=35, n_estimators=100, max_depth=6)
    model.fit(X_train, y_train)
    print(model)
    print(model.feature_importances_)
    plt.barh(X_train.columns, model.feature_importances_)
    plt.show()
    score = model.score(X_train, y_train)
    print("Training score: ", score)

    # # - cross validataion
    score = cross_val_score(model, X, y, cv=10)
    print("Cross Validation Scores are {}".format(score))
    print("Average Cross Validation score :{}".format(score.mean()))
    # print('10-Fold Cross validation:', round(np.mean(cross_val_score(model, X_train, y_train, cv=10)), 2))

    ypred = model.predict(X_test)

    print('Mean Absolute Error:', round(mean_absolute_error(y_test, ypred), 2))
    print('Mean Squared Error:', round(mean_squared_error(y_test, ypred), 2))
    print('Root Mean Squared Error:', round(np.sqrt(mean_squared_error(y_test, ypred)), 2))
    print('R2 square:', round(r2_score(y_test, ypred), 2))

    # x_ax = range(len(y_test))
    # plt.plot(x_ax, y_test, label="original")
    # plt.plot(x_ax, ypred, label="predicted")
    # plt.title("Test and predicted data")
    # plt.legend()
    # plt.show()

    # Fits the explainer
    explainer = shap.Explainer(model.predict, X_train)
    # Calculates the SHAP values - It takes some time
    shap_values = explainer(X_test)
    # Evaluate SHAP values
    shap.plots.bar(shap_values)
