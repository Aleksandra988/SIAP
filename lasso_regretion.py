# use automatically configured the lasso regression algorithm
import warnings

# load the dataset
from numpy import arange
from sklearn.linear_model import LassoCV
from sklearn.model_selection import RepeatedKFold


def fxn():
    warnings.warn("deprecated", DeprecationWarning)

def lasso(dataframe1,dataframe2,dataframe3):
    warnings.simplefilter("ignore")
    fxn()
    print('Beggining lasso regretion:')
    data = dataframe3.values

    print('Data frame shape:')
    print(dataframe3.shape)
    print('Data frame head:')
    print(dataframe3.head)
    #uzima sve podatke i kolonu sa score
    X, y = data[:, 3:], data[:, 2]
    # define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # define model
    model = LassoCV(alphas=arange(0, 1, 0.01), cv=cv, n_jobs=-1)

    print('Model:', model)
    print("X:", X)
    print("y:", y)
    # fit model
    model.fit(X, y)
    # summarize chosen configuration
    print('alpha: %f' % model.alpha_)

    print('Test:',model.predict(data[:,3:]))



