import pandas as pd
# from random_forest_bagging import rf, bagging, rf_regression
from matplotlib import pyplot as plt
from sklearn import model_selection

import lasso_regretion
import random_forest_bagging
import build_dataset
import svm
import xgboost_algorithm
import warnings
import shap


class LazyRegressor:
    pass


if __name__ == '__main__':

    # random_forest_bagging.rf()
    random_forest_bagging.rf_regression()
    # random_forest_bagging.bagging()
    #
    # xgboost_algorithm.xgboost()
    # lasso_regretion.lasso()
    # svm.sss()

    # models = []
    # models.append(('RF', rf_model))
    # models.append(('xgboost', xgboost_model))
    # # models.append(('KNN', KNeighborsClassifier()))
    # # models.append(('CART', DecisionTreeClassifier()))
    # # models.append(('NB', GaussianNB()))
    # models.append(('SVM', svm_model))
    # # evaluate each model in turn
    # results = []
    # names = []
    # scoring = 'accuracy'
    # for name, model in models:
    #     kfold = model_selection.KFold(n_splits=10)
    #     cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    #     results.append(cv_results)
    #     names.append(name)
    #     msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    #     print(msg)
    # # boxplot algorithm comparison
    # fig = plt.figure()
    # fig.suptitle('Algorithm Comparison')
    # ax = fig.add_subplot(111)
    # plt.boxplot(results)
    # ax.set_xticklabels(names)
    # plt.show()