import pandas as pd
import svm

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

    #print(*ret_labels, sep="\n")
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
    #print(df_BLI2018.to_markdown())

    df_BLI2017 = fix_2017_dataset_labels(df_BLI2017)
    return [df_BLI2014, df_BLI2015, df_BLI2016, df_BLI2017, df_BLI2018]
    #df_WHR = read_dataset_by_name('2018')


if __name__ == '__main__':


    # merge all datasets
    bli_datasets = read_BLI_datasets()
    common_labels = find_common_labels(dataset_list=bli_datasets.copy())
    bli_datasets = prune_datasets(common_labels=common_labels, dataset_list=bli_datasets)
    bli_complete = pd.concat(bli_datasets)
    bli_complete.reset_index(inplace=True)
    bli_complete.drop(['index'], axis=1, inplace=True)
    bli_complete.sort_index(axis=1, inplace=True)
    df1_filtered = bli_complete.loc[bli_complete['Country'].isin(
        ['South Africa', 'Russia', 'Brazil', 'Colombia', 'Israel', 'Mexico', 'Iceland', 'Turkey', 'Sweden',
         'Switzerland', 'Lithuania', 'Czech Republic', 'Spain', 'Japan', 'Korea', 'Denmark', 'Chile', 'Latvia',
         'Luxembourg'])]

    #print(df1_filtered.to_markdown())
    #print(bli_complete.to_markdown())
    #print_dataset_labels(bli_datasets)
    print()
    df1 = bli_complete
    list_of_unique_countries = set(df1['Country'].tolist())
    list_of_imputed_datasets_per_country = []
    for country in list_of_unique_countries:
        if any(country in c for c in ['Non-OECD Economies', 'Colombia', 'Lithuania']):
            continue
        list_of_imputed_datasets_per_country.append(svm.solve_missing_values_knn(df1, country))

    bli_imputed_complete = pd.concat(list_of_imputed_datasets_per_country)
    bli_imputed_complete.reset_index(inplace=True)
    bli_imputed_complete.drop(['index'], axis=1, inplace=True)
    bli_imputed_complete.sort_index(axis=1, inplace=True)
    print(bli_imputed_complete.to_markdown())
    '''
    df1 = pd.DataFrame(df1)
    df1.set_axis(labels=df1_labels, axis=1, inplace=True)
    #print(df1.index.is_unique)
    #df1 = df1.drop(['Country'], axis=1)
    #print(df1_country)
    #print(df1['Country'] == None, 'no country column found')
    df1['Country'] = df1_country
    df1.insert(0, 'Country', df1.pop('Country'))
    df1_filtered = df1.loc[df1['Country'].isin(['South Africa', 'Russia', 'Brazil', 'Colombia', 'Israel', 'Mexico', 'Iceland', 'Turkey', 'Sweden', 'Switzerland', 'Lithuania', 'Czech Republic', 'Spain', 'Japan', 'Korea', 'Denmark', 'Chile', 'Latvia', 'Luxembourg'])]
    print(df1_filtered.to_markdown())
    '''