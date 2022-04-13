import pandas as pd
# from random_forest_bagging import rf, bagging, rf_regression
import lasso_regretion
import random_forest_bagging
import build_dataset
import xgboost_algorithm
import warnings


if __name__ == '__main__':

    random_forest_bagging.rf()
    random_forest_bagging.rf_regression()
    random_forest_bagging.bagging()

    xgboost_algorithm.xgboost()
    lasso_regretion.lasso()
