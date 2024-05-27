import pickle
import pandas as pd
from utils_func import *
import os
import time

def benchmarkTraining(Name):
    """
    Name: the name of regions
    """
    # create a dict to save results
    resultsDict = dict()
    load_path = f'./02-electricity-data/XYtables/{Name}.pkl'
    with open(load_path,'rb') as f:
        res_dict = pickle.load(f)
    if Name == 'price':
        ifPrice = True
    else:
        ifPrice = False

    # train the benchmark models
    t1 = time.time()
    Y_test_lasso,param_lasso = trainLasso(res_dict,ifPrice)
    Y_test_lgb,param_lgb = trainLGB(res_dict,ifPrice)
    Y_test_mlp,param_mlp = trainMLP(res_dict,ifPrice)
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
    
    # # evaluation benchmark models
    # eval_benchmark_df = eval_benchmarks('03-benchmarkResults')
    # eval_benchmark_df.to_csv('03-benchmarkResults/results_rmse_mae.csv')

if __name__ == '__main__':
    # Lasso, LightGBM, MLP as benchmarks
    # with open('./02-electricity-data/XYtables/NamesDict.pkl','rb') as f:
    #     NamesDict = pickle.load(f)
    # Names = NamesDict['more_than_5years']
    # for Name in Names:
    #     print(Name)
        # benchmarkTraining(Name)
    benchmarkTraining('price')


        
        
        