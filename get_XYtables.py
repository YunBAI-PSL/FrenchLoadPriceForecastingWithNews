# -*- coding: utf-8 -*-
"""
Created on Mon May 13 17:23:21 2024

@author: yun.bai
"""

import pandas as pd
import numpy as np
from utils_func import *
from jours_feries_france import JoursFeries
from vacances_scolaires_france import SchoolHolidayDates
from sklearn.preprocessing import StandardScaler
import os
import pickle

def get_holidays_dummies(Name):
    """
    Name: can be France nation, grand regions, and metropolis
    """
    dates = pd.date_range(start='2014-01-01',end='2023-12-31').date
    hol_bank_all,hol_sch_all = [],[]
    holDict_bank,holDict_sch = dict(),dict()
    years = list(range(2014,2024))
    JoursEcole = SchoolHolidayDates()
    for year in years:
        if Name == 'Strasbourg': # Strasbourg is zone Alsace-Moselle for bank holiday
            hols_bank = JoursFeries.for_year(year, zone="Alsace-Moselle")
        else:
            hols_bank = JoursFeries.for_year(year)
        # get bank holidays
        for item in hols_bank.items():
            # item[1] is datetime, item[0] is holiday
            holDict_bank[item[1]] = item[0]
        if Name != 'FranceNation':
            # get school holidays only for regions and cities
            hols_sch = JoursEcole.holidays_for_year(year)
            zone = hol_sch_dict[Name].lower()
            for item in hols_sch.items():
                # item[0] is datetime, item[1] is a dict with zone and holiday
                if hols_sch[item[0]][f'vacances_zone_{zone}']:
                    holDict_sch[item[0]] = hols_sch[item[0]]['nom_vacances']
        
    for i in range(len(dates)):
        if dates[i] in holDict_bank.keys():
            hol_bank_all.append(holDict_bank[dates[i]])
        else:
            hol_bank_all.append('Normal')
        if Name != 'FranceNation':
            if dates[i] in holDict_sch.keys():
                hol_sch_all.append(holDict_sch[dates[i]])
            else:
                hol_sch_all.append('Normal')
    
    holidayDf = pd.DataFrame()
    if Name != 'FranceNation':
        holidayDf['Date'],holidayDf['bank_holiday'],holidayDf['school_holiday'] = dates,hol_bank_all,hol_sch_all
    else:
        holidayDf['Date'],holidayDf['bank_holiday'] = dates,hol_bank_all
 
    holidayDf.set_index('Date')

    df_bank = pd.get_dummies(holidayDf['bank_holiday'])
    df_bank['Date'] = dates
    df_bank.set_index('Date')
    if Name != 'FranceNation':
        df_school = pd.get_dummies(holidayDf['school_holiday'])
        df_school['Date'] = dates
        df_school.set_index('Date')   
        df = df_bank.merge(df_school,left_on='Date',right_on='Date',how='left')
        df.drop(columns=['Normal_y'], inplace=True) 
        df.rename(columns={'Normal_x': 'Normal'}, inplace=True)
        df = resampleDataFrame(df)
        df = df.dropna(axis=0)
        df = df.replace({False: 0, True: 1})
        return df
    else:
        df_bank = resampleDataFrame(df_bank)
        df_bank = df_bank.dropna(axis=0)
        df_bank = df_bank.replace({False: 0, True: 1})   
        return df_bank

def get_temperatures(Name):
    """
    Name: can be 'Nation', grand regions, and metropolis
    """
    Temp = pd.read_csv(f'./02-electricity-data/temperatures/All/{Name}_temperature.csv')
    Temp.columns = ['Date','temp']
    Temp = Temp.drop_duplicates(subset=['Date'])
    Temp = resampleDataFrame(Temp)  
    return Temp 

def get_temperatures_nation():
    # get France national temperatures by averaging the temperatures in all the 12 regions,
    # once get the .csv file, the nation temperatures can be get by func get_temperatures('Nation')
    regions = ['Auvergne-Rhone-Alpes','Bourgogne-Franche-Comte','Bretagne',
               'Centre-Val-de-Loire','Grand-Est','Hauts-de-France',
               'Ile-de-France','Normandie','Nouvelle-Aquitaine','Occitanie',
               'Pays-de-la-Loire','Provence-Alpes-Cote-dAzur']
    temp_1 = get_temperatures('Auvergne-Rhone-Alpes')
    temp_arrays = np.zeros(shape=(len(temp_1),len(regions)))
    for i,region in enumerate(regions):
        temp = get_temperatures(region)
        temp_arrays[:,i] = np.array(temp).reshape(-1,)
    nation_mean = np.mean(temp_arrays,axis=1)
    temp_1['temp'] = nation_mean
    temp_1.to_csv('././02-electricity-data/temperatures/All/Nation_temperature.csv')

def addFeats(Name):
    """
    Name: can be 'FranceNation', grand regions, and metropolis. If Name='price', then get price data
    """
    if Name != 'price':
        # load data
        df = pd.read_csv(f'./02-electricity-data/loads/processed_{Name}_load.csv')
        df = df[['Date','Load']]
        df = df.drop_duplicates(subset=['Date'])
        df = resampleDataFrame(df)
    else:
    # price data
        df = pd.read_csv('./02-electricity-data/prices/spot_price_FR_2014_2024_yun_V2.csv')
        df = df.drop_duplicates(subset=['date'])
        df = df.rename(columns={'date':'Date'})
        df['Date'] = pd.to_datetime(df['Date'],utc=True)
        df = df.set_index('Date')
        df = df[df.index>='2018-04-27']  # the wind and solar data are only from this date
        weather_load_gas = df[['wind','solar','CONSOMMATION','PREVISION_J','prix_gaz']]
        weather_load_gas = weather_load_gas.rename(columns={'CONSOMMATION':'generation',
                                                            'PREVISION_J':'load_feat',
                                                            'prix_gaz':'gas_price'})
        
  
    # make the table of 24 hour columns
    df = reshapeDf(df,colName='L1-Hour',if_set_df_index=False)
    
    # shift data to get the D-1 and D-6 lags
    for h in [1,6]:
        for i,col in enumerate(df.columns):
            if 'L1' in col:
                df[f'L{h+1}-Hour{i}'] = df[f'L1-Hour{i}'].shift(h)

    # split issued_date and target_date, and get target_hours (day-ahead hours) as Y
    # temperature is on issued_date, holidays and other time indicators are on target_date
    X_cols = list(df.columns)
    Y_cols = [f'target_Hour_{i}' for i in range(24)]
    df.index = pd.to_datetime(df.index,utc=True)
    df['issued_date'] = df.index.date
    df['target_date'] = df['issued_date'].shift(-1)  # -1 is for day ahead forecasting
    for i in range(24):
        df[f'target_Hour_{i}'] = df[f'L1-Hour{i}'].shift(-1)
    
    # Add temperature on issued date
    if Name != 'price':
        Temp = pd.read_csv(f'./02-electricity-data/temperatures/All/{Name}_temperature.csv')
        Temp = reshapeDf(Temp,colName='Temp',if_set_df_index=True)
        df = df.merge(Temp,left_on='date', right_on='date', how='left')
        X_cols = X_cols+list(Temp.columns)
    
    # Add holiday on target date
    if Name == 'price':
        hols = get_holidays_dummies('FranceNation')
    else:
        hols = get_holidays_dummies(Name)
    hols = hols[hols.index.hour==0]
    hols['date'] = hols.index.shift(-1)  # get the holidays day ahead
    hols.set_index('date',inplace=True) 
    df = df.merge(hols,left_on='date', right_on='date', how='left')
    X_cols = X_cols+list(hols.columns)
    
    # add 0-1 weekends or weekday, 0 for weekends on target date
    df['target_date'] = pd.to_datetime(df['target_date'])
    df['Weekday'] = df['target_date'].dt.weekday
    weekIdx = []
    for i in range(len(df['Weekday'])):
        if df['Weekday'][i]<5:
            weekIdx.append(1) 
        else:
            weekIdx.append(0)
    df['Weekday'] = weekIdx
    X_cols = X_cols+['Weekday']

    # add one-hot encoder for weekday
    df_weekday = pd.get_dummies(df['target_date'].dt.weekday)
    df_weekday.columns = ['Sun','Mon','Tue','Wed','Thu','Fir','Sat']
    df_weekday = df_weekday.replace({True: 1, False: 0})
    df = df.merge(df_weekday,left_on='date', right_on='date', how='left')
    X_cols = X_cols+list(df_weekday.columns)
    
    # add sin doy on target date
    df = encodeTimeStamp(df,isTarget=True)
    X_cols = X_cols + ['dayofyear_sin']
    
    # if forecasting prices, adding wind, solar, load forecasts at target day (D+1); 
    # gas prices lagged 1 day (D); 
    # generate lagged 2 days(D-1) and 10 first hours on the lagged 1 day(D)
    if Name == 'price':
        wind = reshapeDf(weather_load_gas[['wind']],colName='wind',if_set_df_index=False)
        wind = wind.shift(-1) # get with target day
        solar = reshapeDf(weather_load_gas[['solar']],colName='solar',if_set_df_index=False)
        solar = solar.shift(-1) # get with target day
        load_feat = reshapeDf(weather_load_gas[['load_feat']],colName='load_feat',if_set_df_index=False)
        load_feat = load_feat.shift(-1) # get with target day

        gas_price = reshapeDf(weather_load_gas[['gas_price']],colName='gas_price',if_set_df_index=False)
        gas_price = gas_price[[gas_price.columns[0]]] # only keep one gas price, becasue of daily data
        # gas_price is on the issue day
        generation = reshapeDf(weather_load_gas[['generation']],colName='generation',if_set_df_index=False)
        generation_2d = generation.shift(1)  # get the 24 hours for day D-1
        generation_2d.columns = [c+'_2d' for c in generation_2d.columns]
        generation_d = generation.iloc[:,:10] # get the first 10 hour for day D
        generation_d.columns = [c+'_d' for c in generation_d.columns]
        
        df = df.merge(wind,left_on='date', right_on='date', how='left')
        df = df.merge(solar,left_on='date', right_on='date', how='left')
        df = df.merge(load_feat,left_on='date', right_on='date', how='left')
        df = df.merge(gas_price,left_on='date', right_on='date', how='left')
        df = df.merge(generation_2d,left_on='date', right_on='date', how='left')
        df = df.merge(generation_d,left_on='date', right_on='date', how='left')
        
        X_cols = X_cols + list(wind.columns) + list(solar.columns) + \
            list(load_feat.columns) + list(gas_price.columns)+\
            list(generation_2d.columns) + list(generation_d.columns)
    
    df = df.dropna()
    return df,X_cols,Y_cols

def getTrainTest(Name):
    df,X_cols,Y_cols = addFeats(Name)
    # in case using of LightGBM model, the model can not recognise regular name
    X_cols_lgb,Y_cols_lgb = [f'X{i}' for i in range(len(X_cols))],[f'Y{i}' for i in range(len(Y_cols))]
    X,Y = df[X_cols],df[Y_cols]
    X.columns,Y.columns = X_cols_lgb,Y_cols_lgb
    X_train, X_test = X[X.index < '2023-01-01'],X[X.index >= '2023-01-01']
    Y_train, Y_test = Y[Y.index < '2023-01-01'],Y[Y.index >= '2023-01-01']
    
    scaler_x = StandardScaler()
    scaler_x.fit(X_train)
    scaler_y = StandardScaler()
    scaler_y.fit(Y_train)
    
    X_train = scale_data(X_cols_lgb,scaler_x,X_train)
    X_test = scale_data(X_cols_lgb,scaler_x,X_test)
    
    Y_train = scale_data(Y_cols_lgb,scaler_y,Y_train)
    # Y_test = scale_data(Ycolumns,scaler_y,Y_test)
    res_dict = {'data':[X_train,X_test,Y_train,Y_test],
                'scalers':[scaler_x,scaler_y],
                'cols':[X_cols,Y_cols]}
    return res_dict

def getNames():
    # get names of nation(1), regions(12), metropoles(21), 34 in total
    folder_path = './02-electricity-data/loads'
    # check the data set with year 2023 to keep same
    file_paths = [[os.path.join(folder_path, f),f.split('_')[1]] 
                  for f in os.listdir(folder_path) if f.endswith('.csv')]
    Names_more_than_5years,Names_5years,Names_3years = [],[],[]
    for fp in file_paths:
        df = pd.read_csv(fp[0])
        df['Date'] = pd.to_datetime(df['Date'])
        if df['Date'].iloc[-1].year == 2023 or df['Date'].iloc[-1].year == 2024:
            Names_more_than_5years.append(fp[1])
        elif df['Date'].iloc[0].year == 2017:
            Names_5years.append(fp[1])
        else:
            Names_3years.append(fp[1])

    # 23 places was found by Names_more_than_5years
    # National: FranceNation
    # Regions: Bourgogne-Franche-Comte, Pays-de-la-Loire, Hauts-de-France, Provence-Alpes-Cote-dAzur,
            # Bretagne, Normandie, Occitanie, Grand-Est, Auvergne-Rhone-Alpes, 
            # Ile-de-France, Centre-Val-de-Loire, Nouvelle-Aquitaine
    # Metropoles: Brest, Toulouse, Nancy, Montpellier, Grenoble, Rouen, Rennes, Bordeaux,
            # Nice, Strasbourg
    NamesDict = dict()
    NamesDict['more_than_5years'] = Names_more_than_5years
    NamesDict['5years'], NamesDict['3years'] = Names_5years, Names_3years
    with open('./02-electricity-data/XYtables/NamesDict.pkl','wb') as f:
        pickle.dump(NamesDict,f)


if __name__ == '__main__':
    with open('./02-electricity-data/XYtables/NamesDict.pkl','rb') as f:
        NamesDict = pickle.load(f)
    Names = NamesDict['more_than_5years'] + ['price']
    for Name in Names:
        print(Name)
        res_dict = getTrainTest(Name)
        save_path = f'./02-electricity-data/XYtables/{Name}.pkl'
        directory = os.path.dirname(save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(save_path,'wb') as f:
            pickle.dump(res_dict,f)
    
    # save_path = './02-electricity-data/XYtables/price.pkl'
    # with open(save_path,'rb') as f:
    #     res_dict = pickle.load(f)


    
    