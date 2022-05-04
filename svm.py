import shap

import main
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn import svm, metrics
from sklearn.preprocessing import StandardScaler
import build_dataset
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import matplotlib.pyplot as plt


def solve_missing_values_knn(df, country):
    print('country name is ', country, '\n')
    # print(df.to_markdown())
    # df1_labels = df1.columns.values
    # df1_new = df1_new.sort_index(axis=1)
    # df1.set_axis(labels=df1_labels, axis=1, inplace=True)
    # df1['Country'] = df1_country
    # df1.insert(0, 'Country', df1.pop('Country'))
    # set up KNN imputer
    imputer = KNNImputer(missing_values=np.nan, n_neighbors=3, weights="uniform")
    # print(df.to_markdown())
    # country = what country we are imputing for (use past 4 years to impure 5th if it's missing)
    df1_per_country = df.loc[df['Country'] == country]
    if not df1_per_country.isnull().values.any():
        print('no imputation needed')
        return df1_per_country
    # print(df1_per_country.to_markdown())
    # read country column so we can add it later
    df1_countries = df1_per_country['Country']
    # print('****', df1_countries, '****')
    # drop string column for KNN
    df1_per_country = df1_per_country.sort_index(axis=1)
    df1_per_country = df1_per_country.drop(['Country'], axis=1)
    df1_per_country_labels = df1_per_country.columns.values

    x = df1_per_country.values  # returns a numpy array
    df1_new = pd.DataFrame(x)
    df1_new = df1_new.sort_index(axis=1)
    df1_per_country = imputer.fit_transform(df1_per_country)
    df1_per_country = pd.DataFrame(df1_per_country)

    # reset labels
    # print('shape of imputed dataframe ', str(df1_per_country.shape))
    # print('shape of new labels ', str(len(df1_per_country_labels)))
    # print(*df1_per_country_labels, sep='\n')
    # print(df1_per_country.head().to_markdown())
    df1_per_country.set_axis(labels=df1_per_country_labels, axis=1, inplace=True)
    num_of_rows = len(df1_per_country.index)
    df1_per_country['Country'] = [country] * num_of_rows
    # drop and readd as first column
    df1_per_country.insert(0, 'Country', df1_per_country.pop('Country'))
    # print(df1_per_country.to_markdown())

    return df1_per_country


def draw_plot(x_test, y_test, preds, label):
    plt.scatter(x_test, y_test, color='magenta')
    plt.scatter(x_test, preds, color='green')
    plt.title('Happiness analysis (rating= f(' + label + ')')
    plt.xlabel(label)
    plt.ylabel('Rating')
    plt.show()


def svr():
    df, countries = build_dataset.build_imputed_dataset()
    X = df[[i for i in df.columns.tolist() if i != 'Rating' and i != 'Country']]
    X_labels = X.columns.values
    y = df['Rating']
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X_saved, y_saved = X, y
    X = sc_X.fit_transform(X)

    y = np.array(y).reshape((len(y), 1))
    y = sc_y.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=35)
    regr = svm.SVR()
    regr.fit(X_train, y_train)
    preds = regr.predict(X_test)
    preds = np.array(preds).reshape((len(preds), 1))
    preds = sc_y.inverse_transform(preds)
    print(regr.score(X_train, y_train))
    X_test = sc_X.inverse_transform(X_test)
    x_axis = X_test[:, 9]
    # x_axis = np.array(x_axis).reshape((len(x_axis), 1))
    print('x_test shape', x_axis.shape)
    # x_axis = sc_X.inverse_transform(x_axis)
    print('X shape, y shape ', X.shape, y.shape)
    y_pred = regr.predict(X_test)
    print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test, y_pred), 2))
    print('Mean Squared Error:', round(metrics.mean_squared_error(y_test, y_pred), 2))
    print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), 2))
    print('R2 square:', round(metrics.r2_score(y_test, y_pred), 2))

    # Cross validation
    score = cross_val_score(regr, X_train, y_train, cv=10)
    print("Cross Validation Scores are {}".format(score))
    print("Average Cross Validation score :{}".format(score.mean()))

    # SHAP values
    X_train = pd.DataFrame(X_train, columns=X_labels)
    X_test = pd.DataFrame(X_test, columns=X_labels)
    explainer = shap.Explainer(regr.predict, X_train)
    # Calculates the SHAP values - It takes some time
    shap_values = explainer(X_test)
    # Evaluate SHAP values
    shap.plots.bar(shap_values)

    # Grid Search for hyperparameters
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'poly', 'sigmoid']}
    grid = GridSearchCV(regr, param_grid, refit=True, verbose=2)
    grid.fit(X_train, y_train)
    print(grid.best_params_)
    print("Mean cross-validated training accuracy score:", round(grid.best_score_, 2))

