import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from numpy import absolute
from scipy.stats import loguniform
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score, KFold, GridSearchCV, \
    RandomizedSearchCV
from xgboost import XGBClassifier, XGBRegressor
import seaborn as sns

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
    model = XGBRegressor(use_label_encoder=False, random_state=35, n_estimators=100, n_jobs=-1)
    model.fit(X_train, y_train)

    print(model.feature_importances_)
    # plt.barh(X_train.columns, model.feature_importances_)
    # plt.show()

    # # - cross validataion
    score = cross_val_score(model, X, y, cv=10)
    print("Cross Validation Scores are {}".format(score))
    print("Average Cross Validation score :{}".format(score.mean()))

    ypred = model.predict(X_test)

    print('Mean Absolute Error:', round(mean_absolute_error(y_test, ypred), 2))
    print('Mean Squared Error:', round(mean_squared_error(y_test, ypred), 2))
    print('Root Mean Squared Error:', round(np.sqrt(mean_squared_error(y_test, ypred)), 2))
    print('R2 square:', round(r2_score(y_test, ypred), 2))

    # Grid Search for best hyperparameters
    parameters = {'min_child_weight': [1, 2, 3],
                  'gamma': [0, .1, 1, 2],
                  'subsample': [.7, .8, .9, 1],
                  'colsample_bytree': [.8, .9, 1],
                  'max_depth': [2, 3, 4, 5, 7, 10, 12],
                  'n_estimators': [100, 120, 200]}

    xgb_grid = GridSearchCV(model,
                            parameters,
                            cv=10,
                            n_jobs=-1,
                            verbose=True)

    xgb_grid.fit(X_train, y_train)

    print("Optimal hyperparameter combination:", xgb_grid.best_params_)
    print("Mean cross-validated training accuracy score:", xgb_grid.best_score_)

    x_ax = range(len(y_test))
    plt.plot(x_ax, y_test, label="original")
    plt.plot(x_ax, ypred, label="predicted")
    plt.title("RANDOM FOREST - Test and predicted data")
    plt.legend()
    plt.show()

    #SHAP values
    # Fits the explainer
    explainer = shap.Explainer(model.predict, X_train)
    # Calculates the SHAP values - It takes some time
    shap_values = explainer(X_test)
    # Evaluate SHAP values
    shap.plots.bar(shap_values)
