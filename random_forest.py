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

    training, testing, training_labels, testing_labels = train_test_split(X, y, test_size=.25, random_state=42)

    # Normalize the data
    sc = StandardScaler()
    # normed_train_data = pd.DataFrame(sc.fit_transform(training), columns=X.columns)
    # normed_test_data = pd.DataFrame(sc.fit_transform(testing), columns=X.columns)
    clf = RandomForestClassifier()
    clf.fit(training, training_labels)
    preds = clf.predict(testing)
    print(clf.score(training, training_labels))
    print(clf.score(testing, testing_labels))
    print("Accuracy:", metrics.accuracy_score(testing_labels, preds))
    # print(classification_report(testing_labels, clf.predict(testing)))

    # BLI.csv dataset - adding column rating from df3(2018.csv) to df1(BLI.csv) dataset
    df = pd.merge(df1, drop_column(df3)[['Country or region', 'rating']], left_on='Country', right_on='Country or region', how='left')
    df['rating'] = df['rating'].fillna(0)

    XX = df[[i for i in df.columns.tolist() if i != 'Country' and i != 'Country or region' and i != 'rating']]
    # print(X)
    yy = df['rating']

    trainingg, testingg, training_labelss, testing_labelss = train_test_split(XX, yy, test_size=.25, random_state=42)
    rfc = RandomForestClassifier()
    rfc.fit(trainingg, training_labelss)
    predss = rfc.predict(testingg)
    print(rfc.score(trainingg, training_labelss))
    print(rfc.score(testingg, testing_labelss))
    print("Accuracy:", metrics.accuracy_score(testing_labelss, predss))

    # HSL.csv dataset - adding column rating from df3(2018.csv) to df1(HSL.csv) dataset
    dff = pd.merge(df2, drop_column(df3)[['Country or region', 'rating']], left_on='Country',
                   right_on='Country or region', how='left')
    # print(df)
    dff['rating'] = dff['rating'].fillna(0)
    XXX = dff[[i for i in dff.columns.tolist() if i != 'Country' and i != 'Country or region' and i != 'rating']]
    # print(X)
    yyy = dff['rating']

    traininggg, testinggg, training_labelsss, testing_labelsss = train_test_split(XXX, yyy, test_size=.25,
                                                                                  random_state=42)
    # print(training_labels)
    rff = RandomForestClassifier()
    rff.fit(traininggg, training_labelsss)
    predsss = rff.predict(testinggg)
    print(rff.score(traininggg, training_labelsss))
    print(rff.score(testinggg, testing_labelsss))
    print("Accuracy:", metrics.accuracy_score(testing_labelsss, predsss))


def drop_column(df3):
    return df3.drop(['Overall rank', 'Score', 'GDP per capita', 'Social support', 'Healthy life expectancy',
                     'Freedom to make life choices', 'Generosity', 'Perceptions of corruption'], axis=1)
