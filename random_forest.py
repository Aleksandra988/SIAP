import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def rf(df1, df2, df3):
    # 2018.csv dataset
    # rating_pctile = np.percentile(df3['Score'], [3, 4, 5, 6, 7])
    # print(rating_pctile)
    df3['rating'] = 0

    # df3['rating'] = np.where(df3['Score'] < rating_pctile[0], 1, df3['rating'])
    # df3['rating'] = np.where((df3['Score'] >= rating_pctile[0]) & (df3['Score'] <= rating_pctile[1]), 2, df3['rating'])
    # df3['rating'] = np.where((df3['Score'] >= rating_pctile[1]) & (df3['Score'] <= rating_pctile[2]), 3, df3['rating'])
    # df3['rating'] = np.where((df3['Score'] >= rating_pctile[2]) & (df3['Score'] <= rating_pctile[3]), 4, df3['rating'])
    # df3['rating'] = np.where(df3['Score'] > rating_pctile[3], 5, df3['rating'])

    df3['rating'] = np.where(df3['Score'] < 3, 1, df3['rating'])
    df3['rating'] = np.where((df3['Score'] >= 3) & (df3['Score'] <= 4), 2, df3['rating'])
    df3['rating'] = np.where((df3['Score'] >= 4) & (df3['Score'] <= 5), 3, df3['rating'])
    df3['rating'] = np.where((df3['Score'] >= 5) & (df3['Score'] <= 6), 4, df3['rating'])
    df3['rating'] = np.where((df3['Score'] >= 6) & (df3['Score'] <= 7), 5, df3['rating'])
    df3['rating'] = np.where(df3['Score'] > 7, 6, df3['rating'])

    X = df3[[i for i in df3.columns.tolist() if i != 'Overall rank' and i != 'Country or region' and i != 'rating']]
    y = df3['rating']

    training, testing, training_labels, testing_labels = split_dataset(X, y)

    # Normalize the data
    # sc = StandardScaler()
    # normed_train_data = pd.DataFrame(sc.fit_transform(training), columns=X.columns)
    # normed_test_data = pd.DataFrame(sc.fit_transform(testing), columns=X.columns)
    clf = RandomForestClassifier(random_state=137)
    clf.fit(training, training_labels)
    preds = clf.predict(testing)
    print(clf.score(training, training_labels))
    print(clf.score(testing, testing_labels))
    print("Accuracy:", metrics.accuracy_score(testing_labels, preds))
    # print(classification_report(testing_labels, clf.predict(testing)))

    # BLI.csv dataset - adding column rating from df3(2018.csv) to df1(BLI.csv) dataset
    df = pd.merge(df1, drop_column(df3)[['Country or region', 'rating']], left_on='Country',
                  right_on='Country or region', how='left')
    df['rating'] = df['rating'].fillna(df['rating'].median().round(1))

    X2 = df[[i for i in df.columns.tolist() if i != 'Country' and i != 'Country or region' and i != 'rating']]
    # print(X)
    y2 = df['rating']

    training2, testing2, training_labels2, testing_labels2 = split_dataset(X2, y2)
    clf.fit(training2, training_labels2)
    preds2 = clf.predict(testing2)
    print(clf.score(training2, training_labels2))
    print(clf.score(testing2, testing_labels2))
    print("Accuracy:", metrics.accuracy_score(testing_labels2, preds2))

    # HSL.csv dataset - adding column rating from df3(2018.csv) to df1(HSL.csv) dataset
    dff = pd.merge(df2, drop_column(df3)[['Country or region', 'rating']], left_on='Country', right_on='Country or region', how='left')
    dff['rating'] = dff['rating'].fillna(dff['rating'].median().round(1))
    X3 = dff[[i for i in dff.columns.tolist() if i != 'Country' and i != 'Country or region' and i != 'rating']]
    y3 = dff['rating']

    training3, testing3, training_labels3, testing_labels3 = split_dataset(X3, y3)
    clf.fit(training3, training_labels3)
    preds3 = clf.predict(testing3)
    print(clf.score(training3, training_labels3))
    print(clf.score(testing3, testing_labels3))
    print("Accuracy:", metrics.accuracy_score(testing_labels3, preds3))


def drop_column(df3):
    return df3.drop(['Overall rank', 'Score', 'GDP per capita', 'Social support', 'Healthy life expectancy',
                     'Freedom to make life choices', 'Generosity', 'Perceptions of corruption'], axis=1)


def split_dataset(X, y):
    return train_test_split(X, y, test_size=.25, random_state=35)
