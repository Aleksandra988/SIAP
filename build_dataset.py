import pandas as pd


def read_dataset_by_name(name):
    df = pd.read_csv('data-set/' + name + '.csv', na_values=' ')
    return df

# removing as pct, in .. from 2017 data
def fix_2017_dataset_labels(df):

    df.rename(columns={'Years in education in yrs': 'Years in education'}, inplace=True)

    for col in df.columns.values:
        new_label = ''
        # Years in education in yrs was the only one that was messing up so I just skipped it like this
        if ' in ' in col and col != 'Years in education':
            new_label = col.split(' in ')[0]

        elif ' as ' in col:
            new_label = col.split(' as ')[0]
        else:
            new_label = col
        df.rename(columns={col: new_label}, inplace=True)

    return df

def print_dataset_labels(dataset_list):
    list_of_labels_lists = []
    print('14, 15, 16, 17, 18')
    for dsl in dataset_list:
        list_of_labels_lists.append(dsl.columns)

    #zelimo da dobijemo
    # labela1_2018, labela1_2017, labela1_2016
    # labela2_2018, labela2_2017... da uporedimo jesu li iste (ne moraju biti kompletno identicne samo da je isto znacenje)
    printString = ''
    for i in range(len(list_of_labels_lists[0])):
        # item = label_list
        for item in list_of_labels_lists:
            print(item[i])

        print()


def find_common_labels(dataset_list):
    ret_labels = []
    ds_2018 = dataset_list[-1]
    ds_2018.rename(columns={'Househol net wealth': 'Household net financial wealth'}, inplace=True)
    dataset_list.pop()
    for label in ds_2018.columns.values:
        # da li svi poseduju ovu labelu
        allContain = True
        for ds in dataset_list:
            if label not in ds.columns.values:
                allContain = False

        if allContain:
            ret_labels.append(label)

    print(*ret_labels, sep="\n")
    return ret_labels

def prune_datasets(common_labels, dataset_list):
    for ds in dataset_list:
        for label in ds.columns.values:
            if label not in common_labels:
                ds.drop(columns=label, axis=1, inplace=True)
    return dataset_list


def equalize_dataset_labels(dataset_list):
    pass


def read_BLI_datasets():
    #df_HSL = read_dataset_by_name('HSL')
    df_BLI2014 = read_dataset_by_name('BLI2014')
    df_BLI2015 = read_dataset_by_name('BLI2015')
    df_BLI2016 = read_dataset_by_name('BLI2016')
    df_BLI2017 = read_dataset_by_name('BLI2017')
    df_BLI2018 = read_dataset_by_name('BLI2018')

    df_BLI2017 = fix_2017_dataset_labels(df_BLI2017)
    return [df_BLI2014, df_BLI2015, df_BLI2016, df_BLI2017, df_BLI2018]
    #df_WHR = read_dataset_by_name('2018')


if __name__ == '__main__':

    bli_datasets = read_BLI_datasets()
    common_labels = find_common_labels(dataset_list=bli_datasets.copy())
    bli_datasets = prune_datasets(common_labels=common_labels, dataset_list=bli_datasets)
    bli_complete = pd.concat(bli_datasets)
    print(bli_complete.to_markdown())
    #print_dataset_labels(bli_datasets)
