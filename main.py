import pandas as pd
from random_forest_bagging import rf, bagging


def read_dataset():
    df_bli = pd.read_csv('data-set/BLI.csv', na_values=' ')
    print('hey')
    print(df_bli.head())
    df_hsl = pd.read_csv('data-set/HSL.csv')
    df_2018 = pd.read_csv('data-set/2018.csv')
    df_bli2014 = pd.read_csv('data-set/BLI2014.csv')
    df_bli2015 = pd.read_csv('data-set/BLI2015.csv')
    df_bli2016 = pd.read_csv('data-set/BLI2016.csv')
    return df_bli, df_hsl, df_2018, df_bli2014, df_bli2015, df_bli2016


def fill_missing_values(df):
    return df.fillna(df.median().round(1))


def delete_row_col_with_missing_values(df):
    # deleting rows with 60% missing values
    col_number = len(df.columns) * .60
    # print((len(df22.index)))
    df = df[df.isnull().sum(axis=1) < col_number]
    # print((len(df22.index)))

    # deleting cols with 60% missing values
    pct_null = df.isnull().sum() / len(df)
    missing_features = pct_null[pct_null > 0.60].index
    # print(len(df22.columns))
    df = df.drop(missing_features, axis=1)
    # print(len(df22.columns))

    return df.reset_index()


if __name__ == '__main__':


    #df1, df2 = solve_missing_values(df1, df2)

    df1, df2, df3, df_bli2014, df_bli2015, df_bli2016 = read_dataset()
    df2 = delete_row_col_with_missing_values(df2)
    df1 = fill_missing_values(df1)
    df2 = fill_missing_values(df2)
    df3 = fill_missing_values(df3)

    rf(df1, df2, df3)
    bagging(df1, df2, df3)

