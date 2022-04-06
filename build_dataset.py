import pandas as pd


def read_dataset_by_name(name):
    df = pd.read_csv('data-set/' + name + '.csv', na_values=' ')
    return df


if __name__ == '__main__':

    df_HSL = read_dataset_by_name('HSL')
    df_BLI = read_dataset_by_name('BLI')
    df_WHR = read_dataset_by_name('2018')

    print('--------------------HSL---------------------')
    print(*df_HSL.columns.values, sep="\n")
    print()
    print('--------------------BLI---------------------')

    print(*df_BLI.columns.values, sep="\n")
    print()
    print('--------------------2018---------------------')

    print(*df_WHR.columns.values, sep="\n")