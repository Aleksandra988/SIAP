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
from sklearn.model_selection import train_test_split

def lasso():
    warnings.simplefilter("ignore")
    dataset = pd.read_csv('data-set/result.csv')
    dataset = random_forest_bagging.adding_column_class(dataset)
    # print(df1.columns)

    X = dataset.drop(['Rating', 'Country', 'class', 'Rating'], axis=1)
    y = dataset['class']
    # define model evaluation method
    y_pred = lasso_regretion(X, y)


def lasso_regretion(X, y):

    print('------------LASSO REGRETION-------------')
    cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)
    # define model
    model = LassoCV(alphas=arange(0, 1, 0.01), cv=cv, n_jobs=-1)

    X_train, X_test, y_train, y_test = split_dataset(X, y)
    model.fit(X_train, y_train)

    print('Score:', model.score(X_train,y_train))
    y_pred = model.predict(X_test)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', numpy.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    errors = abs(y_pred - y_test)
    mape = 100 * (errors / y_test)
    # Calculate and display accuracy
    accuracy = 100 - numpy.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')
    print(model.score(X_test, y_test))

    x_ax = range(len(y_test))
    plt.plot(x_ax, y_test, label="original")
    plt.plot(x_ax, y_pred, label="predicted")
    plt.title("LASSO REGRETION-Test and predicted data")
    plt.legend()
    plt.show()

    return  y_pred

def split_dataset(X, y):
    return train_test_split(X, y, test_size=0.25, random_state=100)