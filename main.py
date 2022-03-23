import pandas as pd


def read_dataset():
    df_bli = pd.read_csv('data-set/BLI.csv')
    df_hsl = pd.read_csv('data-set/HSL.csv')
    return df_bli, df_hsl


def solve_missing_values(df11, df22):
    df11 = df11.fillna(df11.median().round(1))
    # print(df1)

    # deleting rows with 60% missing values
    col_number = len(df22.columns) * .60
    print((len(df22.index)))
    df22 = df22[df22.isnull().sum(axis=1) < col_number]
    print((len(df22.index)))

    # deleting cols with 70% missing values
    pct_null = df22.isnull().sum() / len(df22)
    missing_features = pct_null[pct_null > 0.70].index
    print(len(df22.columns))
    df22 = df22.drop(missing_features, axis=1)
    print(len(df22.columns))

    df22 = df22.reset_index()
    df22 = df22.fillna(df22.median().round(1))
    return df11, df22


if __name__ == '__main__':
    df1, df2 = read_dataset()
    df1, df2 = solve_missing_values(df1, df2)
