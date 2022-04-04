import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def rf(df1, df2, df3):
    print('------------RANDOM FOREST-------------')
    # 2018.csv dataset
    df3 = adding_column_rating(df3)

    X = df3[[i for i in df3.columns.tolist() if i != 'Overall rank' and i != 'Country or region' and i != 'rating']]
    y = df3['rating']
    rf_training_and_prediction(X, y, '2018.csv')

    # BLI.csv dataset - adding column rating from df3(2018.csv) to df1(BLI.csv) dataset
    df = merge_two_datasets(df1, df3)
    df['rating'] = df['rating'].fillna(df['rating'].median().round(1))
    X2 = df[[i for i in df.columns.tolist() if i != 'Country' and i != 'Country or region' and i != 'rating']]
    y2 = df['rating']
    rf_training_and_prediction(X2, y2, "BLI.csv")

    # HSL.csv dataset - adding column rating from df3(2018.csv) to df1(HSL.csv) dataset
    dff = merge_two_datasets(df2, df3)
    dff['rating'] = dff['rating'].fillna(dff['rating'].median().round(1))
    X3 = dff[[i for i in dff.columns.tolist() if i != 'Country' and i != 'Country or region' and i != 'rating']]
    y3 = dff['rating']
    rf_training_and_prediction(X3, y3, "HSL.csv")


def adding_column_rating(df3):
    df3['rating'] = 0
    df3['rating'] = np.where(df3['Score'] < 4, 1, df3['rating'])
    # df3['rating'] = np.where((df3['Score'] >= 3) & (df3['Score'] <= 4), 2, df3['rating'])
    df3['rating'] = np.where((df3['Score'] >= 4) & (df3['Score'] <= 5), 2, df3['rating'])
    df3['rating'] = np.where((df3['Score'] >= 5) & (df3['Score'] <= 6), 3, df3['rating'])
    df3['rating'] = np.where((df3['Score'] >= 6) & (df3['Score'] <= 7), 4, df3['rating'])
    df3['rating'] = np.where(df3['Score'] > 7, 5, df3['rating'])
    return df3


def merge_two_datasets(df, df3):
    df3 = df3.drop(['Overall rank', 'Score', 'GDP per capita', 'Social support', 'Healthy life expectancy',
                    'Freedom to make life choices', 'Generosity', 'Perceptions of corruption'], axis=1)
    return pd.merge(df, df3[['Country or region', 'rating']], left_on='Country',
                    right_on='Country or region', how='left')


def split_dataset(X, y):
    return train_test_split(X, y, test_size=.25, random_state=35)


def rf_training_and_prediction(X, y, dataset):
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    # Normalize the data
    # sc = StandardScaler()
    # normed_train_data = pd.DataFrame(sc.fit_transform(training), columns=X.columns)
    # normed_test_data = pd.DataFrame(sc.fit_transform(testing), columns=X.columns)
    clf = RandomForestClassifier(random_state=137)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print(clf.score(X_train, y_train))
    # print(clf.score(testing, testing_labels))
    print(dataset + " Accuracy:", metrics.accuracy_score(y_test, preds))


def bagging(df1, df2, df3):
    print('------------BAGGING-------------')
    df3 = adding_column_rating(df3)

    # 2018.csv dataset
    X = df3[[i for i in df3.columns.tolist() if i != 'Overall rank' and i != 'Country or region' and i != 'rating']]
    y = df3['rating']
    bagging_training_and_prediction(X, y, '2018.csv')

    # BLI.csv dataset - adding column rating from df3(2018.csv) to df1(BLI.csv) dataset
    df = merge_two_datasets(df1, df3)
    df['rating'] = df['rating'].fillna(df['rating'].median().round(1))
    X2 = df[[i for i in df.columns.tolist() if i != 'Country' and i != 'Country or region' and i != 'rating']]
    y2 = df['rating']
    bagging_training_and_prediction(X2, y2, "BLI.csv")

    # HSL.csv dataset - adding column rating from df3(2018.csv) to df1(HSL.csv) dataset
    dff = merge_two_datasets(df2, df3)
    dff['rating'] = dff['rating'].fillna(dff['rating'].median().round(1))
    X3 = dff[[i for i in dff.columns.tolist() if i != 'Country' and i != 'Country or region' and i != 'rating']]
    y3 = dff['rating']
    bagging_training_and_prediction(X3, y3, "HSL.csv")


def bagging_training_and_prediction(X, y, dataset_name):
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    print('Bagging with random forest classifier:')
    bag_model_rf = BaggingClassifier(RandomForestClassifier(), random_state=137)
    bag_model_rf.fit(X_train, y_train)
    # print(bag_model.oob_score_)
    print(bag_model_rf.score(X_train, y_train))
    preds = bag_model_rf.predict(X_test)
    print(dataset_name + " Accuracy:", metrics.accuracy_score(y_test, preds))

    print('Bagging with KN classifier:')
    bag_model_kn = BaggingClassifier(KNeighborsClassifier(), random_state=137)
    bag_model_kn.fit(X_train, y_train)
    # print(bag_model.oob_score_)
    print(bag_model_kn.score(X_train, y_train))
    preds = bag_model_kn.predict(X_test)
    print(dataset_name + " Accuracy:", metrics.accuracy_score(y_test, preds))

    print()
    # scores = cross_val_score(RandomForestClassifier(n_estimators=50), X, y)
    # scores.mean()
