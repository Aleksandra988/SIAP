import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def rf(df1, df2, df3):
    # 2018.csv dataset
    df3['rating'] = 0
    df3['rating'] = np.where(df3['Score'] < 3, 1, df3['rating'])
    df3['rating'] = np.where((df3['Score'] >= 3) & (df3['Score'] <= 4), 2, df3['rating'])
    df3['rating'] = np.where((df3['Score'] >= 4) & (df3['Score'] <= 5), 3, df3['rating'])
    df3['rating'] = np.where((df3['Score'] >= 5) & (df3['Score'] <= 6), 4, df3['rating'])
    df3['rating'] = np.where((df3['Score'] >= 6) & (df3['Score'] <= 7), 5, df3['rating'])
    df3['rating'] = np.where(df3['Score'] > 7, 6, df3['rating'])

    X = df3[[i for i in df3.columns.tolist() if i != 'Overall rank' and i != 'Country or region' and i != 'rating']]
    y = df3['rating']
    training_and_prediction(X, y, '2018.csv')

    # BLI.csv dataset - adding column rating from df3(2018.csv) to df1(BLI.csv) dataset
    df = merge_two_datasets(df1, df3)
    df['rating'] = df['rating'].fillna(df['rating'].median().round(1))
    X2 = df[[i for i in df.columns.tolist() if i != 'Country' and i != 'Country or region' and i != 'rating']]
    y2 = df['rating']
    training_and_prediction(X2, y2, "BLI.csv")

    # HSL.csv dataset - adding column rating from df3(2018.csv) to df1(HSL.csv) dataset
    dff = merge_two_datasets(df2, df3)
    dff['rating'] = dff['rating'].fillna(dff['rating'].median().round(1))
    X3 = dff[[i for i in dff.columns.tolist() if i != 'Country' and i != 'Country or region' and i != 'rating']]
    y3 = dff['rating']
    training_and_prediction(X3, y3, "HSL.csv")


def merge_two_datasets(df, df3):
    df3 = df3.drop(['Overall rank', 'Score', 'GDP per capita', 'Social support', 'Healthy life expectancy',
                    'Freedom to make life choices', 'Generosity', 'Perceptions of corruption'], axis=1)
    return pd.merge(df, df3[['Country or region', 'rating']], left_on='Country',
                    right_on='Country or region', how='left')


def training_and_prediction(X, y, dataset):
    training, testing, training_labels, testing_labels = train_test_split(X, y, test_size=.25, random_state=35)
    # Normalize the data
    # sc = StandardScaler()
    # normed_train_data = pd.DataFrame(sc.fit_transform(training), columns=X.columns)
    # normed_test_data = pd.DataFrame(sc.fit_transform(testing), columns=X.columns)
    clf = RandomForestClassifier(random_state=137)
    clf.fit(training, training_labels)
    preds = clf.predict(testing)
    print(clf.score(training, training_labels))
    # print(clf.score(testing, testing_labels))
    print(dataset + " Accuracy:", metrics.accuracy_score(testing_labels, preds))
