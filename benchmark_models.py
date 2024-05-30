import pickle
import pandas as pd
from utils_func import *
import os
import time
os.environ['PYTHONWARNINGS'] = 'ignore'

def benchmarkTraining(Name):
    """
    Name: the name of regions
    """
    # create a dict to save results
    resultsDict = dict()
    load_path = f'./02-electricity-data/XYtables/{Name}.pkl'
    with open(load_path,'rb') as f:
        res_dict = pickle.load(f)
    # if Name == 'price' or 'FranceNation':
    # # if Name == 'price':
    #     ifPrice = True
    # else:
    #     ifPrice = False
    ifPrice = True
    print(ifPrice)
    # train the benchmark models
    t1 = time.time()
    Y_test_lasso,best_params = trainLasso(res_dict,ifPrice)
    # Y_test_lgb,param_lgb = trainLGB(res_dict,ifPrice)
    # Y_test_mlp,param_mlp = trainMLP(res_dict,ifPrice)
    resultsDict['lasso_fore'], resultsDict['lasso_param'] = Y_test_lasso,best_params
    # resultsDict['lgb_fore'], resultsDict['lgb_param'] = Y_test_lgb,param_lgb
    # resultsDict['mlp_fore'], resultsDict['mlp_param'] = Y_test_mlp,param_mlp 
    t2 = time.time()
    # resultsDict['lasso_fore'].to_csv(f'./03-benchmarkResults/{Name}/NoText/recalibrate_lasso.csv')
    print(t2-t1)

    # save the results dict
    save_path = f'./03-benchmarkResults/{Name}/NoText/recalibrate_lasso.pkl'
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(save_path,'wb') as f:
        pickle.dump(resultsDict,f)

    
    # # evaluation benchmark models
    # eval_benchmark_df = eval_benchmarks('03-benchmarkResults')
    # eval_benchmark_df.to_csv('03-benchmarkResults/results_rmse_mae.csv')
    # return Y_test_lasso,best_params

if __name__ == '__main__':
    # Lasso, LightGBM, MLP as benchmarks
    with open('./02-electricity-data/XYtables/NamesDict.pkl','rb') as f:
        NamesDict = pickle.load(f)
    Names = NamesDict['more_than_5years']+['price']
    # print(Names)
    for Name in Names[5:]:
        print(Name)
        benchmarkTraining(Name)

    # Y_test_lasso,best_params = benchmarkTraining('FranceNation')
    # eval_benchmark_df = eval_benchmarks('03-benchmarkResults')
    # # eval_benchmark_df.to_csv('03-benchmarkResults/results_rmse_mae.csv')
    
    # with open('03-benchmarkResults/FranceNation/NoText/lasso_lgb_mlp.pkl','rb') as f:
    #     lasso_lgb_mlp = pickle.load(f)
    
    # eval_lasso = evaluate_deterministic(lasso_lgb_mlp['lasso_fore']).mean()
    # eval_lgb = evaluate_deterministic(lasso_lgb_mlp['lgb_fore']).mean()
    # eval_mlp = evaluate_deterministic(lasso_lgb_mlp['mlp_fore']).mean()

    # # eval national load
    # with open('02-electricity-data/XYtables/FranceNation.pkl','rb') as f:
    #     FranceNationLoad = pickle.load(f)
    # df = pd.read_csv('./02-electricity-data/loads/processed_FranceNation_load_J1.csv')
    # df = df[['Date','Load','Day-ahead forecast']]
    # df = df.drop_duplicates(subset=['Date'])
    # df = resampleDataFrame(df)
    # df = df[df.index.year==2023]

    # eval_nation = evaluate_deterministic(df).mean()
    # rmse,mae = calculate_rmse_mae(df)
    # print(rmse,mae)
    
    # eval recalibrate nation
    # df1 = pd.read_csv('./03-benchmarkResults/FranceNation/NoText/recalibrate_lasso.csv')
    # df1 = df1.drop_duplicates(subset=['Date'])
    # df1 = resampleDataFrame(df1)
    # eval_nation_recali = evaluate_deterministic(df1).mean()

    