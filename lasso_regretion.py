# use automatically configured the lasso regression algorithm
import warnings

# load the dataset
#from numpy import arange
from sklearn.linear_model import LassoCV
from sklearn.model_selection import RepeatedKFold
import build_dataset

def lasso(dataframe1, dataframe2, dataframe3):
    warnings.simplefilter("ignore")
    print('Beggining lasso regretion:')
    data1 = dataframe1.values
    data2 = dataframe2.values
    data3 = dataframe3.values
    print('Data1 frame shape:')
    print(dataframe1.shape)
    print('Data2 frame head:')
    print(dataframe2.head)
    print('Data3 frame head:')
    print(dataframe3.head)

    # uzima sve podatke i kolonu sa score
    X3, y3 = data3[:, 3:], data3[:, 2]
    # define model evaluation method
    model3 = lasso_regretion(X3, y3)
    print("Data 3:",model3.predict(data3[:, 3:]))

    # bolje radi za data1, jer ima vise podataka
    X1, y1 = lasso_prepair(data1, data3)
    # define model evaluation method
    model1 = lasso_regretion(X1, y1)
    print("Data 1:",model1.predict(data1[:, 1:]))

    X2, y2 = lasso_prepair(data2[:,1:], data3)
    # define model evaluation method
    model2 = lasso_regretion(X2, y2)
    print("Data 2:",model2.predict(data2[:, 2:]))




def lasso_prepair(data1, data2):
    X, y = [], []
    for d1 in data1:
        for d2 in data2:
            if d1[0] == d2[1]:
                X.append(d1[1:])
                y.append(d2[2])
                break
    print(X)
    print(y)
    print(len(X), ',', len(y))
    return X, y


def lasso_regretion(X, y):
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # define model
    model = LassoCV(alphas=build_dataset.np(0, 1, 0.01), cv=cv, n_jobs=-1)

    model.fit(X, y)
    # summarize chosen configuration
    print('alpha: %f' % model.alpha_)
    return model

if __name__ == '__main__':
    bli_imputed_complete, countries = build_dataset.build_imputed_dataset
    print(bli_imputed_complete)