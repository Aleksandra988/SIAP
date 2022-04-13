import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
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
    model.fit(X_train, y_train, verbose=True)
    # make predictions for test data
    # y_pred = model.predict(X_test)
    # predictions = [round(value) for value in y_pred]
    # # evaluate predictions
    print(model.score(X_test, y_test))
    # y_pred = model.predict(X_test)
    # predictions = [round(value) for value in y_pred]
    # # evaluate predictions
    # accuracy = accuracy_score(y_test, predictions)
    # print("Accuracy: %.2f%%" % (accuracy * 100.0))
    # accuracy = accuracy_score(y_test, predictions)
    # print("Accuracy: %.2f%%" % (accuracy * 100.0))

    # # BLI.csv dataset - adding column rating from df3(2018.csv) to df1(BLI.csv) dataset
    # df = random_forest_bagging.merge_two_datasets_regression(df1, df3)
    # df['Score'] = df['Score'].fillna(df['Score'].median().round(1))
    # X2 = df[[i for i in df.columns.tolist() if i != 'Country' and i != 'Country or region' and i != 'Score']]
    # y2 = df['Score']
    # # print(df.isnull())
    # # print(df['Personal earnings'])
    # X_train, X_test, y_train, y_test = random_forest_bagging.split_dataset(X2, y2)
    # # fit model no training data
    # model1 = XGBRegressor(use_label_encoder=False)
    # model1.fit(X_train, y_train)
    # # make predictions for test data
    # y_pred = model1.predict(X_test)
    # predictions = [round(value) for value in y_pred]
    # # evaluate predictions
    # print(model1.score(X_test, y_test))
    #
    # # HSL.csv dataset - adding column rating from df3(2018.csv) to df1(HSL.csv) dataset
    # dff = random_forest_bagging.merge_two_datasets_regression(df2, df3)
    # dff = dff[[i for i in dff.columns.tolist() if i != 'Country' and i != 'Country or region']]
    # dff['Score'] = dff['Score'].fillna(dff['Score'].median().round(1))
    # x = dff.values  # returns a numpy array
    # min_max_scaler = preprocessing.MinMaxScaler()
    # x_scaled = min_max_scaler.fit_transform(x)
    # dff = pd.DataFrame(x_scaled)
    # print(dff)
    # X3 = dff[[i for i in dff.columns.tolist() if i != 'Country' and i != 'Country or region' and i != 'Score']]
    # y3 = dff['Score']
    # X_train, X_test, y_train, y_test = random_forest_bagging.split_dataset(X3, y3)
    # # fit model no training data
    # model2 = XGBRegressor(use_label_encoder=False)
    # model2.fit(X_train, y_train)
    # # make predictions for test data
    # y_pred = model2.predict(X_test)
    # predictions = [round(value) for value in y_pred]
    # # evaluate predictions
    # print(model2.score(X_test, y_test))
