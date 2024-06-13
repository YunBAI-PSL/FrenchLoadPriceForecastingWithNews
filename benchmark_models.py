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
    # Name ='price'
    resultsDict = dict()
    load_path = f'./02-electricity-data/XYtables/{Name}.pkl'
    with open(load_path,'rb') as f:
        res_dict = pickle.load(f)
    # if Name == 'price' or 'FranceNation':
    # # if Name == 'price':
    #     ifPrice = True
    # else:
    #     ifPrice = False
    ifRecalibrate = True
    print(ifRecalibrate)

    ifEcoData = False
    print(ifEcoData)
    # train the benchmark models
    t1 = time.time()
    Y_test_lasso,best_params = trainLasso(Name, res_dict,ifRecalibrate,ifEcoData)
    # Y_test_lgb,param_lgb = trainLGB(res_dict,ifRecalibrate)
    # Y_test_mlp,param_mlp = trainMLP(res_dict,ifRecalibrate)
    # Y_test_etr,param_etr = trainETR(res_dict,ifRecalibrate)
    
    resultsDict['lasso_fore'], resultsDict['lasso_param'] = Y_test_lasso,best_params
    # resultsDict['lgb_fore'], resultsDict['lgb_param'] = Y_test_lgb,param_lgb
    # resultsDict['mlp_fore'], resultsDict['mlp_param'] = Y_test_mlp,param_mlp 
    # resultsDict['etr_fore'], resultsDict['etr_param'] = Y_test_etr,param_etr
    t2 = time.time()
    # resultsDict['lasso_fore'].to_csv(f'./03-benchmarkResults/{Name}/NoText/recalibrate_lasso.csv')
    print(t2-t1)

    # save the results dict
    save_path = f'./03-benchmarkResults/{Name}/NoText/recalibrate_lasso_test_2021_day_all_data.pkl'
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(save_path,'wb') as f:
        pickle.dump(resultsDict,f)

    # evaluation benchmark models
    # eval_benchmark_df = eval_benchmarks('03-benchmarkResults')
    # eval_benchmark_df.to_csv('03-benchmarkResults/results_rmse.csv')
    return Y_test_lasso,best_params

if __name__ == '__main__':
    # Lasso, LightGBM, MLP as benchmarks
    # with open('./02-electricity-data/XYtables/NamesDict.pkl','rb') as f:
    #     NamesDict = pickle.load(f)
    # Names = NamesDict['more_than_5years']+['price']
    # # print(Names)
    # rmse_l,mae_l = [],[]
    # for Name in Names:
    #     print(Name)
        # Y_test_lasso,best_params = benchmarkTraining(Name)
        # get persistence and skill scores
        # persi_df = calPersistence(Name)
        # persi_df.to_csv(f'./03-benchmarkResults/{Name}/NoText/persi_df.csv')
        # eval and save
        # with open(f'./03-benchmarkResults/{Name}/NoText/lasso_lgb_mlp.pkl', 'rb') as f:
        #     resultsDict = pickle.load(f)
        # lasso_fore = resultsDict['lasso_fore']
        # metrics = evaluate_deterministic(lasso_fore).mean()
        # RMSE,MAE = metrics[0],metrics[1]
        # rmse_l.append(RMSE)
        # mae_l.append(MAE)
    
    # eval_df = pd.DataFrame()
    # eval_df['Locations'] = Names
    # eval_df['persistence_rmse'],eval_df['persistence_mae'] = rmse_l,mae_l
    # eval_df.to_csv('./03-benchmarkResults/results_rmse.csv',index=False)

    Y_test_lasso,best_params = benchmarkTraining('price')
    print(best_params)
    eval_nation_recali = evaluate_deterministic(Y_test_lasso).mean()
    print(eval_nation_recali)
    # eval_benchmark_df = eval_benchmarks('03-benchmarkResults')
    # # eval_benchmark_df.to_csv('03-benchmarkResults/results_rmse_mae.csv')
    
    # with open('03-benchmarkResults/FranceNation/NoText/recalibrate_lasso.pkl','rb') as f:
    #     lasso_lgb_mlp = pickle.load(f)
    
    # eval_lasso = evaluate_deterministic(lasso_lgb_mlp['lasso_fore']).mean()
    # eval_lgb = evaluate_deterministic(lasso_lgb_mlp['lgb_fore']).mean()
    # eval_mlp = evaluate_deterministic(lasso_lgb_mlp['mlp_fore']).mean()
