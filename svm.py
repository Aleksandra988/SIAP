import main
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn import preprocessing
import build_dataset


def solve_missing_values_knn(df, country):
    print('country name is ', country, '\n')
    # df1_labels = df1.columns.values
    # df1_new = df1_new.sort_index(axis=1)
    # df1.set_axis(labels=df1_labels, axis=1, inplace=True)
    # df1['Country'] = df1_country
    # df1.insert(0, 'Country', df1.pop('Country'))
    # set up KNN imputer
    imputer = KNNImputer(missing_values=np.nan, n_neighbors=3, weights="uniform")

    # country = what country we are imputing for (use past 4 years to impure 5th if it's missing)
    df1_per_country = df.loc[df['Country'] == country]
    if not df.isnull().values.any():
        return df1_per_country
    print(df1_per_country.to_markdown())
    # read country column so we can add it later
    df1_countries = df1_per_country['Country']
    print('****', df1_countries, '****')
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
    df1_per_country.set_axis(labels=df1_per_country_labels, axis=1, inplace=True)
    num_of_rows = len(df1_per_country.index)
    df1_per_country['Country'] = [country] * num_of_rows
    # drop and readd as first column
    df1_per_country.insert(0, 'Country', df1_per_country.pop('Country'))
    print(df1_per_country.to_markdown())

    return df1_per_country


if __name__ == '__main__':
    # df1, df2 = main.read_dataset()
    df1 = build_dataset.read_dataset_by_name('BLI')
    df2 = build_dataset.read_dataset_by_name('HSL')
    df1_country = df1['Country']
    print(df1.shape)
    # df1 = df1.drop(['Country'], axis=1)
    df1 = df1.sort_index(axis=1)
    print('0000')
    print(df1.shape)

    df1_labels = df1.columns.values
    print(df1_labels.shape)
    df1 = solve_missing_values_knn(df1)
    # print(len(df1.columns))
    # (df1.columns)
    # print(df1)
    df1 = pd.DataFrame(df1)
    # df1.iloc[0] = df1_labels
    print(df1.columns.values)
    df1.set_axis(labels=df1_labels, axis=1, inplace=True)
    # df1.insert(loc=0, column='Country', value=df1_country)
    df1['Country'] = df1_country
    # f1 = df1.sort_index(axis=1)
    print(df1_labels)
    # df1['Country'] = df1_country
    df1.insert(0, 'Country', df1.pop('Country'))
    print(df1.to_markdown())