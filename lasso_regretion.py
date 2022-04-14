# use automatically configured the lasso regression algorithm
import warnings

# load the dataset
import numpy
from numpy import arange
from sklearn.linear_model import LassoCV
from sklearn.model_selection import RepeatedKFold
import pandas as pd
import random_forest_bagging
from sklearn import metrics
from matplotlib import pyplot as plt

def lasso():
    warnings.simplefilter("ignore")
    dataset = pd.read_csv('data-set/result.csv')
    dataset = random_forest_bagging.adding_column_class(dataset)
    # print(df1.columns)
    X = dataset[[i for i in dataset.columns.tolist() if i != 'Overall rank' and i != 'Country' and i != 'Rating']]
    y = dataset['Rating']
    # define model evaluation method
    y_pred = lasso_regretion(X, y)

    x_ax = range(len(y))
    plt.plot(x_ax, y, label="original")
    plt.plot(x_ax, y_pred, label="predicted")
    plt.title("LASSO REGRETION-Test and predicted data")
    plt.legend()
    plt.show()

def lasso_regretion(X, y):

    print('------------LASSO REGRETION-------------')
    cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)
    # define model
    model = LassoCV(alphas=arange(0, 1, 0.01), cv=cv, n_jobs=-1)

    model.fit(X, y)

    print('Score:', model.score(X,y))
    y_pred = model.predict(X)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y, y_pred))
    print('Root Mean Squared Error:', numpy.sqrt(metrics.mean_squared_error(y, y_pred)))

    errors = abs(y_pred - y)
    mape = 100 * (errors / y)
    # Calculate and display accuracy
    accuracy = 100 - numpy.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')
    print(model.score(X, y))

    return  y_pred