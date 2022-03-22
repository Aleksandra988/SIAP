import pandas as pd


def read_dataset():
    df_bli = pd.read_csv('data-set/BLI.csv')
    df_hsl = pd.read_csv('data-set/HSL.csv')
    return df_bli, df_hsl


def organize_missing_values(df11, df22):
    df11 = df11.fillna(df11.median().round(1))
    # print(df1)SS
    i = 0
    for value in df22.columns:
        if df22[value].isnull().sum() > len(df22.index) / 2:
            df22 = df22.drop(i)
        i = i + 1
    df22 = df22.reset_index()
    df22 = df22.fillna(df22.median().round(1))
    print(df22)
    return df11, df22


if __name__ == '__main__':
    df1, df2 = read_dataset()
    df1, df2 = organize_missing_values(df1, df2)
