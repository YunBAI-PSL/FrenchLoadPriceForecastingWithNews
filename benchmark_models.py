import pickle
import pandas as pd
from utils_func import *
import os
import time

def get_df(Name):
    """
    Name: can be 'FranceNation', grand regions, metropolis, and 'price'
    """
    # get load or price
    if Name != 'price':
        # load data
        df = pd.read_csv(f'./02-electricity-data/loads/processed_{Name}_load.csv')
        df = df[['Date','Load']]
    else:
    # price data
        df = pd.read_csv('./02-electricity-data/prices/National_prices.csv')
        df = df[['Date','price']]
    df = df.drop_duplicates(subset=['Date'])
    df = resampleDataFrame(df)

    # get temperatures locally; for price, get national temperatures
    if Name != 'price':
        Temp = pd.read_csv(f'./02-electricity-data/temperatures/All/{Name}_temperature.csv')
    else:
        Temp = pd.read_csv(f'./02-electricity-data/temperatures/All/FranceNation_temperature.csv')
    Temp.columns = ['Date','temp']
    Temp = Temp.drop_duplicates(subset=['Date'])
    Temp = resampleDataFrame(Temp)  

    df = df.merge(Temp, left_on='Date', right_on='Date', how='left')
    df = df.dropna()

    return df 

if __name__ == '__main__':
    # Lasso, LightGBM, MLP as benchmarks
    with open('./02-electricity-data/XYtables/NamesDict.pkl','rb') as f:
        NamesDict = pickle.load(f)
    Names = NamesDict['more_than_5years']
    for Name in Names:
        print(Name)
        # create a dict to save results
        resultsDict = dict()
        load_path = f'./02-electricity-data/XYtables/{Name}.pkl'
        with open(load_path,'rb') as f:
            res_dict = pickle.load(f)

        # train the benchmark models
        t1 = time.time()
        Y_test_lasso,param_lasso = trainLasso(res_dict)
        Y_test_lgb,param_lgb = trainLGB(res_dict)
        Y_test_mlp,param_mlp = trainMLP(res_dict)
        resultsDict['lasso_fore'], resultsDict['lasso_param'] = Y_test_lasso,param_lasso
        resultsDict['lgb_fore'], resultsDict['lgb_param'] = Y_test_lgb,param_lgb
        resultsDict['mlp_fore'], resultsDict['mlp_param'] = Y_test_mlp,param_mlp 
        t2 = time.time()
        print(t2-t1)

        # save the results dict
        save_path = f'./03-benchmarkResults/{Name}/NoText/lasso_lgb_mlp.pkl'
        directory = os.path.dirname(save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(save_path,'wb') as f:
            pickle.dump(resultsDict,f)
        
