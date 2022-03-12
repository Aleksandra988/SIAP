import pandas as pd

def read_dataset():
    df1 = pd.read_csv('data-set/BLI.csv')
    df2 = pd.read_csv('data-set/HSL.csv')
    # print(df)
    # print(df1.head())
    # print(df2.head())
    print(df1.columns)
    print(df2.columns)
    # print(df['LOCATION'])


if __name__ == '__main__':
    read_dataset()

