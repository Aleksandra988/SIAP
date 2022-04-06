import main
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn import preprocessing


def solve_missing_values_knn(df1_new, df22):
    imputer = KNNImputer(missing_values=np.nan, n_neighbors=2, weights="uniform")

    #df11.drop()
    #df1_new = df11.drop(['Country'], axis=1)
    x = df1_new.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df1_new = pd.DataFrame(x_scaled)
    df1_new = df1_new.sort_index(axis=1)
    df1_new = imputer.fit_transform(df1_new)
    #df1_new['Country'] = df11['Country']
    #imputer.fit_transform(df22)

    return df1_new, df22


if __name__ == '__main__':
    df1, df2 = main.read_dataset()
    df1_country = df1['Country']
    print(df1.shape)
    df1 = df1.drop(['Country'], axis=1)
    df1 = df1.sort_index(axis=1)
    print('0000')
    print(df1.shape)

    df1_labels = df1.columns.values
    print(df1_labels.shape)
    df1, df2 = solve_missing_values_knn(df1, df2)
    #print(len(df1.columns))
    #(df1.columns)
    #print(df1)
    df1 = pd.DataFrame(df1)
    #df1.iloc[0] = df1_labels
    print(df1.columns.values)
    df1.set_axis(labels=df1_labels, axis=1, inplace=True)
    #df1.insert(loc=0, column='Country', value=df1_country)
    df1['Country'] = df1_country
    #f1 = df1.sort_index(axis=1)
    print(df1_labels)
    #df1['Country'] = df1_country
    df1.insert(0, 'Country', df1.pop('Country'))
    print(df1.to_markdown())
