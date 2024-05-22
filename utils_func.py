# -*- coding: utf-8 -*-
"""
Created on Tue May 14 10:28:34 2024

@author: yun.bai
"""
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

#%%
def resampleDataFrame(df_original):
    period = 'H'
    if type(df_original) == pd.core.series.Series:
        df_original = df_original.to_frame()
    if 'Date' in list(df_original.columns):
        df_original['Date'] = pd.to_datetime(df_original['Date'],utc=True)
        
        df_original.index = pd.DatetimeIndex(data=df_original['Date'])
        df_original = df_original.drop(['Date'], axis=1)
    
    df_resampled = df_original.resample(period).mean()
    
    lenDfOrig = len(df_original.index)
    lenDfResamp = len(df_resampled.index)
    if lenDfOrig < lenDfResamp:
        df_resampled = df_original.resample(period).ffill()
    else:
        df_resampled = df_original.resample(period).mean()
        
    return df_resampled

# Dictionary of school holidays in France
# if national load used, school holidays are not considered
hol_sch_dict = {
    'Auvergne-Rhone-Alpes':'A',
    'Bourgogne-Franche-Comte':'A',
    'Bretagne':'B',
    'Centre-Val-de-Loire':'B',
    'Grand-Est':'B',
    'Hauts-de-France':'B',
    'Ile-de-France':'C',
    'Normandie':'B',
    'Nouvelle-Aquitaine':'A',
    'Occitanie':'C',
    'Pays-de-la-Loire':'B',
    'Provence-Alpes-Cote-dAzur':'B',
    'Aix-Marseille':'B',
    'Bordeaux':'A',
    'Brest':'B',
    'Clermont-Ferrand':'A',
    'Dijon':'A',
    'Grand-Lyon':'A',
    'Grand-Paris':'C',
    'Grenoble':'A',
    'Lille':'B',
    'Montpellier':'C',
    'Nancy':'B',
    'Nantes':'B',
    'Nice':'B',
    'Orleans':'B',
    'Rennes':'B',
    'Rouen':'B',
    'Saint-Etienne':'A',
    'Strasbourg':'B',
    'Toulon':'B',
    'Toulouse':'C',
    'Tours':'B'}

def encodeTimeStamp(df,isTarget):
    """
    isTarget: True of False. Indicator to show if use the target date or not. The index of df is issued_date
    """
    if isTarget:
        dayofweek = df['target_date'].dt.dayofweek
        dayofyear = df['target_date'].dt.dayofyear
    else:
        dayofweek = df.index.dayofweek
        dayofyear = df.index.dayofyear
    
    # df['dayofweek_sin'] = np.sin(2 * np.pi * dayofweek / 7.0)
    # df['dayofweek_cos'] = np.cos(2 * np.pi * dayofweek / 7.0)
    
    df['dayofyear_sin'] = np.sin(2 * np.pi * dayofyear / 365.25)
    # df['dayofyear_cos'] = np.cos(2 * np.pi * dayofyear / 365.25)
    
    return df

# reshape the one-col houly data into 24-col data
def reshapeDf(df,colName,if_set_df_index):
    """
    df: csv dataset input, with or without index setting. Only two columns: Date and colName
    colName: interested variable, could be load, Temp, weather
    if_set_df_index: if True, need to set df index
    """
    if if_set_df_index:
        df['Date'] = pd.to_datetime(df['Date'],utc=True)
        df = df.set_index('Date')
        df.fillna(method='ffill', inplace=True)
    df['date'] = df.index.date
    df['hour'] = df.index.hour
    df = df.pivot_table(index='date', columns='hour', values=df.columns[0])
    df.columns = [colName+str(i) for i in df.columns]
    df.index = pd.to_datetime(df.index,utc=True)
    return df

def scale_data(cols,scaler,data):
    """
    func to get scaled dataframe.
    cols: column names of dataframe
    scaler: pre-defined scaler
    data: pd.series or dataframe
    """
    if type(data) == pd.core.series.Series:
        data = data.to_frame()
    index_data = data.index
    data = scaler.transform(data)
    data = pd.DataFrame(data,columns=cols).set_index(index_data)
    return data

def expand_data(data,colName):
    fore_df = pd.DataFrame()
    for i in range(len(data)):
        temp = data.iloc[i,:]
        temp_index = [data.index[i] + datetime.timedelta(hours=j) for j in range(24)]
        temp = temp.to_frame()
        if type(colName) == list:
            temp.columns = colName
        else:
            temp.columns = [colName]
        temp = temp.reset_index(drop=True)
        temp['Date'] = temp_index
        temp = temp.set_index('Date')
        fore_df = pd.concat([fore_df,temp])
    return fore_df

def trainLasso(res_dict):
    """
    res_dict: results from XYtables, {'data':[X_train,X_test,Y_train,Y_test],
                				      'scalers':[scaler_x,scaler_y],
                				      'cols':[X_cols,Y_cols]}
    """
    # load data   
    X_train, X_test, Y_train, Y_test = res_dict['data']
    scaler_x,scaler_y = res_dict['scalers']

    lr = Lasso(random_state=42)
    alphas = {'estimator__alpha': np.logspace(-4, 1, 50)}
    # alphas = {'estimator__alpha': np.array([0.01])}
    multi_lr = MultiOutputRegressor(lr)
    tscv = TimeSeriesSplit(n_splits=5)

    lasso_regressor = GridSearchCV(multi_lr, alphas, 
                                   scoring='neg_mean_squared_error', 
                                   cv=tscv, n_jobs=-1)
    lasso_regressor.fit(X_train, Y_train)
    best_params = lasso_regressor.best_params_
    best_model = lasso_regressor.best_estimator_
    preY = best_model.predict(X_test)
    preY = scaler_y.inverse_transform(preY)
    Y_pres = pd.DataFrame(preY,index=Y_test.index,columns=Y_test.columns)
    Y_pres_all = expand_data(Y_pres,colName='Lasso_forecasts')
    Y_test_all = expand_data(Y_test,colName='Load')
    Y_test_all = Y_test_all.merge(Y_pres_all,left_on='Date', right_on='Date', how='inner')
    
    return Y_test_all, best_params

# Lgb model with optuna
def lgbObjective(X_train,Y_train,trial):
    """
    trial: compute trial by optuna
    """
    tscv = TimeSeriesSplit(n_splits=5)

    params = {
        'objective': 'regression',
        'metric': 'mse',
        'verbosity': -1,
        'random_state': 42,
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'num_leaves': trial.suggest_int('num_leaves', 2, 100)
    }

    # save mse for each fold
    mse_scores = []
    model = MultiOutputRegressor(lgb.LGBMRegressor(**params))
    try:
        for train_index, test_index in tscv.split(X_train):
            print(train_index)
            X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
            y_train_fold, y_test_fold = Y_train.iloc[train_index], Y_train.iloc[test_index]
            
            model.fit(X_train_fold, y_train_fold)
            preds = model.predict(X_test_fold)
            mse = mean_squared_error(y_test_fold, preds)
            mse_scores.append(mse)
        average_mse = np.mean(mse_scores)
    except Exception as e:
        print(f"An error occurred: {e}")
        average_mse = float('inf')
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return average_mse

def trainLGB(res_dict):
    """
    res_dict: results from XYtables, {'data':[X_train,X_test,Y_train,Y_test],
                				      'scalers':[scaler_x,scaler_y],
                				      'cols':[X_cols,Y_cols]}
    """
    X_train, X_test, Y_train, Y_test = res_dict['data']
    scaler_x,scaler_y = res_dict['scalers']

    study = optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: lgbObjective(X_train,Y_train,trial), 
                   n_trials=100,
                   n_jobs=-1,
                   callbacks=[MaxTrialsCallback(n_trials=100, states=(TrialState.COMPLETE, TrialState.PRUNED))])

    best_params = study.best_params
    best_model = MultiOutputRegressor(lgb.LGBMRegressor(**best_params))
    best_model.fit(X_train, Y_train)
    preY = best_model.predict(X_test)
    preY = scaler_y.inverse_transform(preY)
    Y_pres = pd.DataFrame(preY,index=Y_test.index,columns=Y_test.columns)
    Y_pres_all = expand_data(Y_pres,colName='LGB_forecasts')
    Y_test_all = expand_data(Y_test,colName='Load')
    Y_test_all = Y_test_all.merge(Y_pres_all,left_on='Date', right_on='Date', how='inner')
    return Y_test_all, best_params

# MLP model with optuna
def mlpObjective(X_train,Y_train,trial):
    """
    trial: compute trial by optuna
    """
    tscv = TimeSeriesSplit(n_splits=5)

    params = {
        'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(50,), (100,), (50, 50), (100, 100)]),
        'activation': trial.suggest_categorical('activation', ['tanh', 'relu']),
        'solver': trial.suggest_categorical('solver', ['adam']),
        'alpha': trial.suggest_loguniform('alpha', 1e-4, 1e-1),
        'learning_rate_init': trial.suggest_loguniform('learning_rate_init', 1e-4, 1e-1),
        'max_iter': 500  # set higher iter times
    }

    # save mse for each fold
    mse_scores = []
    model = MLPRegressor(**params, random_state=0)
    try:
        for train_index, test_index in tscv.split(X_train):
            X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
            y_train_fold, y_test_fold = Y_train.iloc[train_index], Y_train.iloc[test_index]
            model.fit(X_train_fold, y_train_fold)

            preds = model.predict(X_test_fold)
            mse = mean_squared_error(y_test_fold, preds)
            mse_scores.append(mse)
        average_mse = np.mean(mse_scores)
    except Exception as e:
        print(f"An error occurred: {e}")
        average_mse = float('inf')
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return average_mse

def trainMLP(res_dict):
    """
    res_dict: results from XYtables, {'data':[X_train,X_test,Y_train,Y_test],
                				      'scalers':[scaler_x,scaler_y],
                				      'cols':[X_cols,Y_cols]}
    """
    X_train, X_test, Y_train, Y_test = res_dict['data']
    scaler_x,scaler_y = res_dict['scalers']

    study = optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: mlpObjective(X_train,Y_train,trial), 
                   n_trials=100,
                   n_jobs=-1,
                   callbacks=[MaxTrialsCallback(n_trials=100, states=(TrialState.COMPLETE, TrialState.PRUNED))])

    best_params = study.best_params
    best_model = MLPRegressor(**best_params)
    best_model.fit(X_train, Y_train)
    preY = best_model.predict(X_test)
    preY = scaler_y.inverse_transform(preY)
    Y_pres = pd.DataFrame(preY,index=Y_test.index,columns=Y_test.columns)
    Y_pres_all = expand_data(Y_pres,colName='MLP_forecasts')
    Y_test_all = expand_data(Y_test,colName='Load')
    Y_test_all = Y_test_all.merge(Y_pres_all,left_on='Date', right_on='Date', how='inner')
   
    return Y_test_all, best_params

# evaluation basic funcs
def evaluate_deterministic(Y):
    stringPred = Y.columns[1]
    stringObs = 'Load'
    dateList = Y.index.tolist()
    Y['horizon'] = [dt.date() for dt in dateList]
    
    Y['WEerror'] = Y[stringPred] - Y[stringObs]
    
    Y['WEerrorSquare'] = Y['WEerror']**2
    Y['WEabsError'] = Y['WEerror'].abs()
    Y['WEmapeError'] = Y['WEabsError']/np.abs(Y[stringObs])*100
    
    #Y['OFIIerror'] = Y['ForecastedLoad'] - Y[stringObs]
    #Y['OFIIerrorSquare'] = Y['OFIIerror']**2
    #Y['OFIIabsError'] = Y['OFIIerror'].abs()
    #Y['OFIIsmapeError'] = Y['OFIIabsError']/((Y['ForecastedLoad'] + Y[stringObs])/2)

    dfEvaluationDeterministic = pd.DataFrame(columns=['RMSE', 'MAPE'])
    dfEvaluationDeterministic['RMSE'] = Y.groupby('horizon').mean()['WEerrorSquare']**0.5
    dfEvaluationDeterministic['MAPE'] = Y.groupby('horizon').mean()['WEmapeError']

    return dfEvaluationDeterministic
# %%
