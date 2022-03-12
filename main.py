import pandas as pd

def read_dataset():
    df1 = pd.read_csv('data-set/BLI.csv')
    df2 = pd.read_csv('data-set/HSL.csv')
    # df3 = pd.read_csv('data-set/data.csv')
    df4 = pd.read_csv('data-set/BLI_new.csv')
    df5 = pd.read_csv('data-set/HSL_new.csv')
    # print(df3)
    # print(df1.head())
    print(df4.head())
    print(df5.head())
    # print(df1.columns)
    # print(df2.columns)
    # print(df['LOCATION'])


if __name__ == '__main__':
    read_dataset()

