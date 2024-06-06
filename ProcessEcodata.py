import pandas as pd
from utils_func import resampleDataFrame

def convert_df(df):
    for d in df['Date']:
        if 'T1' in d:
            d = d[:4]+'-01-01'
        elif 'T2' in d:
            d = d[:4]+'-04-01'
        elif 'T3' in d:
            d = d[:4]+'-07-01'
        elif 'T4' in d:
            d = d[:4]+'-10-01'
        else:
            d = d+'-01'
    df['Date'] = pd.to_datetime(df['Date'],utc=True)
    df = resampleDataFrame(df)
    return df

def combine_df():
    GDP = pd.read_excel('02-electricity-data/EconomicData/GDP.xlsx')
    Inflation = pd.read_csv('02-electricity-data/EconomicData/Inflation.csv',sep=';')
    Unemployment = pd.read_excel('02-electricity-data/EconomicData/Unemployment.xlsx')
    GDP = convert_df(GDP)
    Inflation = convert_df(Inflation)
    Unemployment = convert_df(Unemployment)
    eco_data = Unemployment.merge(Inflation,left_on='Date',right_on='Date',how='inner')
    eco_data = eco_data.merge(GDP,left_on='Date',right_on='Date',how='inner')
    eco_data.to_csv('02-electricity-data/EconomicData/eco_data.csv')

combine_df()