import csv

import pandas as pd
import svm
import numpy as np


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
    # print('14, 15, 16, 17, 18')
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


def replace_country_name(ds, old, new):
    index = ds.index[ds['Country'] == old].tolist()
    if index:
        index = index[0]
        ds.at[index, 'Country'] = new
    return ds


def sync_country_names(whr_list):
    year = 2014
    for whr in whr_list:
        # print(year)
        year = year + 1

        whr = replace_country_name(whr, 'South Korea', 'Korea')
        whr = replace_country_name(whr, 'Slovakia', 'Slovak Republic')
        whr = replace_country_name(whr, 'USA', 'United States')

    # print('country sync done')
    return whr_list


def add_happiness_ranking_to_respective_years(bli_datasets):

    wh_2015 = read_dataset_by_name('whr/2015')
    wh_2016 = read_dataset_by_name('whr/2016')
    wh_2017 = read_dataset_by_name('whr/2017')
    wh_2017.rename(columns={'Happiness.Score': 'Happiness Score'}, inplace=True)

    wh_2018 = read_dataset_by_name('whr/2018')
    wh_2018.rename(columns={'Score': 'Happiness Score', 'Country or region': 'Country'}, inplace=True)
    # for now we will use 2015 results for both 2014 and 2015 BLI index cuz we're missing a 2014 WHR
    wh_list = [wh_2015, wh_2016, wh_2017, wh_2018]
    wh_list = sync_country_names(wh_list)
    bli_datasets = sync_country_names(bli_datasets)
    wh_list = [wh_list[0], wh_list[0], wh_list[1], wh_list[2], wh_list[3]]
    unique_countries = set(bli_datasets[0]['Country'].tolist())
    year = 2014
    for index in range(len(bli_datasets)):
        #print(year, ' >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        year = year + 1
        bli = bli_datasets[index]
        whr = wh_list[index]
        bli['Rating'] = np.nan
        for country in unique_countries:
            if any(country in c for c in ['Non-OECD Economies', 'Colombia', 'Lithuania', 'OECD - Total']):
                continue
            #print(country)
            #bli.query('Country == ' + country)['Rating'] = whr.query('Country == ' + country)['Happiness Score']
            rating = whr[whr['Country'] == country]['Happiness Score'].values[0]
            #print(rating)
            index = bli.index[bli['Country'] == country].tolist()[0]
            #print('index is ', index)

            bli.at[index, 'Rating'] = rating
            #print(bli[bli['Country'] == country].to_markdown())
            #df.loc[df.ID == 103, 'FirstName'] = "Matt"
            #bli.loc[bli.Country == country, 'Rating'] = rating
            #bli[bli['Country'] == country]['Rating'] = whr[whr['Country'] == country]['Happiness Score']
            #rslt_df = dataframe[dataframe['Percentage'] > 80]
        bli.drop(bli[bli.Rating.isnull()].index, inplace=True)
    # print(bli_datasets[0].to_markdown())
    return bli_datasets


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


def build_imputed_dataset():

    # merge all datasets
    bli_datasets = read_BLI_datasets()
    common_labels = find_common_labels(dataset_list=bli_datasets.copy())
    bli_datasets = prune_datasets(common_labels=common_labels, dataset_list=bli_datasets)
    bli_datasets = add_happiness_ranking_to_respective_years(bli_datasets)
    # print('happiness added')
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
    # print(list_of_unique_countries)
    for country in list_of_unique_countries:
        if any(country in c for c in ['Non-OECD Economies', 'Colombia', 'Lithuania', 'OECD - Total']):
            continue
        list_of_imputed_datasets_per_country.append(svm.solve_missing_values_knn(df1, country))

    bli_imputed_complete = pd.concat(list_of_imputed_datasets_per_country)
    bli_imputed_complete.reset_index(inplace=True)
    bli_imputed_complete.drop(['index'], axis=1, inplace=True)
    bli_imputed_complete.sort_index(axis=1, inplace=True)
    # print(bli_imputed_complete.to_markdown())
    return bli_imputed_complete, list_of_unique_countries

if __name__ == '__main__':

    bli_imputed_complete, countries = build_imputed_dataset()
    bli_imputed_complete.to_csv(r'data-set\result.csv')
