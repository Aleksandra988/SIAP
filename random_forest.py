import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def rf(df3):
    rating_pctile = np.percentile(df3['Score'], [3, 4, 5, 6])
    df3['rating'] = 0

    df3['rating'] = np.where(df3['Score'] < rating_pctile[0], 1, df3['rating'])
    df3['rating'] = np.where((df3['Score'] >= rating_pctile[0]) & (df3['Score'] <= rating_pctile[1]), 2, df3['rating'])
    df3['rating'] = np.where((df3['Score'] >= rating_pctile[1]) & (df3['Score'] <= rating_pctile[2]), 3, df3['rating'])
    df3['rating'] = np.where((df3['Score'] >= rating_pctile[2]) & (df3['Score'] <= rating_pctile[3]), 4, df3['rating'])
    df3['rating'] = np.where(df3['Score'] > rating_pctile[3], 5, df3['rating'])

    # print(df3['rating'])
    X = df3[[i for i in df3.columns.tolist() if i != 'Overall rank' and i != 'Country or region' and i != 'rating']]
    # print(X)
    y = df3['rating']

    training, testing, training_labels, testing_labels = train_test_split(X, y, test_size=.25, random_state=42)
    # print(training_labels)

    # tr, test = train_test_split(df3, test_size=.25, random_state=42)
    # features = tr.drop(['Overall rank', 'Country or region', 'rating'], axis=1).columns
    # training = tr[features]
    # training_labels = tr['rating']
    # testing = test[features]
    # testing_labels = test['rating']

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