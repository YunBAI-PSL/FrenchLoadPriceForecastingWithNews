# -*- coding: utf-8 -*-
"""
Created on Tue May 14 10:28:34 2024

@author: yun.bai
"""
import pandas as pd
import numpy as np
import datetime
import time
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso,LassoLarsIC
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import ExtraTreesRegressor
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
import glob
import os
import pickle
from scipy.stats import skew, kurtosis, norm, probplot, jarque_bera
# from LEARscaling import scaling
os.environ['PYTHONWARNINGS'] = 'ignore'

class MultiOutputLEAR:
    def __init__(self):
        self.model = None

    def recalibrate(self, Xtrain, Ytrain, columns_hol):
        """Function to recalibrate the LEAR model. 
        
        It uses a training (Xtrain, Ytrain) pair for recalibration
        
        Parameters
        ----------
        Xtrain : numpy.array
            Input in training dataset. It should be of size *[n,m]* where *n* is the number of days
            in the training dataset and *m* the number of input features
        
        Ytrain : numpy.array
            Output in training dataset. It should be of size *[n,24]* where *n* is the number of days 
            in the training dataset and 24 are the 24 prices of each day
        
        columns_hol: list
            used to indicate the 0-1 variables in X, can be holidays and weekends, 
            because 0-1 variables cannot be scaled in this case
        Returns
        -------
        numpy.array
            The prediction of day-ahead prices after recalibrating the model        
        
        """

        # # Applying Invariant, aka asinh-median transformation to the prices
        [Ytrain], self.scalerY = scaling([Ytrain], 'Invariant')

        # # Rescaling all inputs except dummies (7 last features)
        all_columns = set(range(Xtrain.shape[1]))
        not_columns_hol = list(all_columns - set(columns_hol))

        [Xtrain_no_dummies], self.scalerX = scaling([Xtrain[:, not_columns_hol]], 'Invariant')
        Xtrain[:, not_columns_hol] = Xtrain_no_dummies

        self.models = {}
        for h in range(24):

            # Estimating lambda hyperparameter using LARS
            param_model = LassoLarsIC(criterion='aic', max_iter=2500)
            param = param_model.fit(Xtrain, Ytrain[:, h]).alpha_

            # Re-calibrating LEAR using standard LASSO estimation technique
            model = Lasso(max_iter=2500, alpha=param)
            model.fit(Xtrain, Ytrain[:, h])
            # models[h] = model

            self.models[h] = model

    def predict(self, X, columns_hol):
        """Function that makes a prediction using some given inputs.
        
        Parameters
        ----------
        X : numpy.array
            Input of the model.
        
        Returns
        -------
        numpy.array
            An array containing the predictions.
        """

        # Predefining predicted prices
        Yp = np.zeros((X.shape[0], 24))

        # # Rescaling all inputs except dummies (7 last features)
        all_columns = set(range(X.shape[1]))
        not_columns_hol = list(all_columns - set(columns_hol))
        
        X_no_dummies = self.scalerX.transform(X[:, not_columns_hol])
        X[:, not_columns_hol] = X_no_dummies

        # Predicting the current date using a recalibrated LEAR
        for h in range(24):

            # Predicting test dataset and saving
            Yp[:,h] = self.models[h].predict(X)
        
        Yp = self.scalerY.inverse_transform(Yp)

        return Yp

    def recalibrate_predict(self, Xtrain, Ytrain, Xtest, columns_hol):
        """Function that first recalibrates the LEAR model and then makes a prediction.

        The function receives the training dataset, and trains the LEAR model. Then, using
        the inputs of the test dataset, it makes a new prediction.
        
        Parameters
        ----------
        Xtrain : numpy.array
            Input of the training dataset.
        Xtest : numpy.array
            Input of the test dataset.
        Ytrain : numpy.array
            Output of the training dataset.
        
        Returns
        -------
        numpy.array
            An array containing the predictions in the test dataset.
        """

        self.recalibrate(Xtrain=Xtrain, Ytrain=Ytrain, columns_hol=columns_hol)    

        Yp = self.predict(X=Xtest,columns_hol=columns_hol)

        return Yp
    
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

# forecasting models
def calPersistence(Name):
    load_path = f'./02-electricity-data/XYtables/{Name}.pkl'
    with open(load_path,'rb') as f:
        res_dict = pickle.load(f)
    _, _, Y_train, Y_test = res_dict['data']
    _,scaler_y = res_dict['scalers']
    idx_Y,cols_Y = Y_train.index,Y_train.columns
    Y_train = scaler_y.inverse_transform(Y_train)
    Y_train = pd.DataFrame(Y_train,index=idx_Y,columns=cols_Y)
    Y = pd.concat([Y_train,Y_test])
    Y_persi = Y.shift(1)
    Y_persi = Y_persi[Y_persi.index >= '2023-12-01']
    Y_pres_all = expand_data(Y_persi,colName='Persi_forecasts')
    Y_test_all = expand_data(Y_test,colName='Load')
    Y_test_all = Y_test_all.merge(Y_pres_all,left_on='Date', right_on='Date', how='inner')
    return Y_test_all

def rolling_train_price(Name,X_train,Y_train,X_test,Y_test,best_model,scaler_x,scaler_y):
    X_train_new = X_train.copy()
    Y_train_new = Y_train.copy()
    X_test_new = X_test.copy()
    Y_test_new = Y_test.copy()  
    
    all_predictions = []  # List to store all predictions
    sunday_indexes = X_test_new.resample('W-SUN').last().index
    # sunday_indexes = X_test_new.resample('M').last().index
    params_dict = dict()

    # for sunday_index in sunday_indexes:
    for sunday_index in list(X_test_new.index):
        t3 = time.time()
        print('*'*10)
        print(sunday_index)
        # Predict with the model until next Sunday
        try:
            preY_new = best_model.predict(X_test_new.loc[:sunday_index])
            preY_new = scaler_y.inverse_transform(preY_new)
        except:
            break
            
        X_train_idx,X_test_idx = X_train_new.index,X_test_new.index
        Y_train_idx,Y_cols = Y_train_new.index, Y_train.columns
        X_cols = X_train_new.columns
        X_train_new,X_test_new = scaler_x.inverse_transform(X_train_new),scaler_x.inverse_transform(X_test_new)
        Y_train_new = scaler_y.inverse_transform(Y_train_new)

        X_train_new = pd.DataFrame(X_train_new,index=X_train_idx,columns=X_cols)
        X_test_new = pd.DataFrame(X_test_new,index=X_test_idx,columns=X_cols)
        Y_train_new = pd.DataFrame(Y_train_new,index=Y_train_idx,columns=Y_cols)
    
        # Save predictions
        all_predictions.append(pd.DataFrame(preY_new, index=X_test_new.loc[:sunday_index].index, columns=Y_test_new.columns))
        
        # Add data until next Sunday to training set
        if Name != 'price':
            try:
                X_train_new = pd.concat([X_train_new.iloc[-365*3:], X_test_new.loc[:sunday_index]])
                Y_train_new = pd.concat([Y_train_new.iloc[-365*3:], Y_test_new.loc[:sunday_index]])
            except:
                X_train_new = pd.concat([X_train_new, X_test_new.loc[:sunday_index]])
                Y_train_new = pd.concat([Y_train_new, Y_test_new.loc[:sunday_index]])
        else:
            X_train_new = pd.concat([X_train_new, X_test_new.loc[:sunday_index]])
            Y_train_new = pd.concat([Y_train_new, Y_test_new.loc[:sunday_index]])
                
        # Remove data until next Sunday from test set
        if X_test_new.loc[sunday_index + pd.Timedelta(days=1):].shape[0] == 0:
            break
        else:
            X_test_new = X_test_new.loc[sunday_index + pd.Timedelta(days=1):]
            Y_test_new = Y_test_new.loc[sunday_index + pd.Timedelta(days=1):]

        # rescale data
        scaler_x = StandardScaler()
        scaler_x.fit(X_train_new)
        scaler_y = StandardScaler()
        scaler_y.fit(Y_train_new)

        X_train_new = scale_data(X_cols,scaler_x,X_train_new)
        X_test_new = scale_data(X_cols,scaler_x,X_test_new)
        Y_train_new = scale_data(Y_cols,scaler_y,Y_train_new)

        # retrain models
        best_params,best_model = LassoCore(X_train_new, Y_train_new)
        params_dict[sunday_index] = best_params
        t4 = time.time()
        print('*'*10)
        print(t4-t3)
    
    # Combine all predictions into a single DataFrame
    Y_pres_all = pd.concat(all_predictions)
    Y_pres_all.columns = [f'Lasso_forecasts_{col}' for col in Y_pres_all.columns]
    Y_pres_all.drop_duplicates(inplace=True)
    Y_pres_all = expand_data(Y_pres_all,colName='Lasso_forecasts')
    Y_test_all = expand_data(Y_test,colName='Price')
    # Merge predictions with original test data
    Y_test_all = Y_test_all.merge(Y_pres_all,left_on='Date', right_on='Date', how='inner')
    return Y_test_all,params_dict

def LassoCore(X_train, Y_train):
    lr = Lasso(random_state=42)
    alphas = {'estimator__alpha': np.logspace(-4, 1, 50)}
    # alphas = {'estimator__alpha': np.array([0.1])}
    multi_lr = MultiOutputRegressor(lr)
    tscv = TimeSeriesSplit(n_splits=5)

    lasso_regressor = GridSearchCV(multi_lr, alphas, 
                                   scoring='neg_mean_squared_error', 
                                   cv=tscv, n_jobs=-1)
    lasso_regressor.fit(X_train, Y_train)
    best_params = lasso_regressor.best_params_
    best_model = lasso_regressor.best_estimator_
    return best_params,best_model

def get_inverse_scale_df(df,scaler):
    cols,idx = df.columns,df.index
    df = scaler.inverse_transform(df)
    df = pd.DataFrame(df,index=idx,columns=cols)
    return df

def add_Ecodata(X_train,X_test,scaler_x):
    # inverse the data to add eco features
    X_train = get_inverse_scale_df(X_train,scaler_x)
    X_test = get_inverse_scale_df(X_test,scaler_x)

    # add eco features
    eco_data = pd.read_csv('02-electricity-data/EconomicData/eco_data.csv')
    eco_data = resampleDataFrame(eco_data)

    X_train = X_train.rename_axis('Date')
    X_test = X_test.rename_axis('Date')
    X_train = X_train.merge(eco_data,left_on='Date',right_on='Date',how='inner')
    X_test = X_test.merge(eco_data,left_on='Date',right_on='Date',how='inner')

    X_cols = X_train.columns
    
    # for feat in X_cols:
    #     X_train[feat] = X_train[feat].apply(lambda x: np.log(x+1))
    #     X_test[feat] = X_test[feat].apply(lambda x: np.log(x+1))
    
    scaler_x = StandardScaler()
    scaler_x.fit(X_train)
    
    X_train = scale_data(X_cols,scaler_x,X_train)
    X_test = scale_data(X_cols,scaler_x,X_test)
    
    return X_train,X_test,scaler_x

def extract_feature_importance(best_model, feature_names):
    feature_importance = np.array([est.coef_ for est in best_model.estimators_])
    avg_feature_importance = np.mean(np.abs(feature_importance), axis=0)
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': list(avg_feature_importance)
    })
    
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
    
    return feature_importance_df

def trainLasso(Name,res_dict,ifRecalibrate,ifEcoData):
    """
    res_dict: results from XYtables, {'data':[X_train,X_test,Y_train,Y_test],
                				      'scalers':[scaler_x,scaler_y],
                				      'cols':[X_cols,Y_cols]}
    ifRecalibrate: True of False, indicator for Recalibrate forecasting or not
    ifEcoData: True of False, indicator for adding economic data or not
    """
    # load data   
    X_train, X_test, Y_train, Y_test = res_dict['data']
    scaler_x,scaler_y = res_dict['scalers']
    # to predict the prices in 2021,2022,2023
    X_train_p = X_train[X_train.index<'2021-01-01']
    X_test_p = X_train[(X_train.index>='2021-01-01')&(X_train.index<'2022-01-01')]
    Y_train_p = Y_train[Y_train.index<'2021-01-01']
    Y_test_p = Y_train[(Y_train.index>='2021-01-01')&(Y_train.index<'2022-01-01')]
    X_train, X_test, Y_train, Y_test = X_train_p, X_test_p, Y_train_p, Y_test_p

    # if X_train.shape[0] > 365*3:
    #     X_train = X_train.iloc[-365*3:,:]
    #     Y_train = Y_train.iloc[-365*3:,:]

    if ifEcoData:
        X_train,X_test,scaler_x = add_Ecodata(X_train,X_test,scaler_x)

    best_params,best_model = LassoCore(X_train, Y_train)

    # get feature importance
    # feature_names = res_dict['cols'][0] + ['Unemployment','Inflation','GDP']
    # feature_names = res_dict['cols'][0]
    # feature_importance_df = extract_feature_importance(best_model, feature_names)
    # feature_importance_df.to_csv('03-benchmarkResults/FranceNation/NoText/feat_importance.csv',index=False)

    # get best params from the saved file
    # with open('03-benchmarkResults/FranceNation/NoText/lasso_lgb_mlp.pkl','rb') as f:
    #     lasso_lgb_mlp = pickle.load(f)
    # best_params = lasso_lgb_mlp['lasso_param']['estimator__alpha']
    # best_model = MultiOutputRegressor(Lasso(alpha=best_params, random_state=42))
    # best_model.fit(X_train,Y_train)

    if ifRecalibrate:
        Y_test_all,params_dict = rolling_train_price(Name,X_train,Y_train,X_test,Y_test,best_model,scaler_x,scaler_y)
        return Y_test_all, [best_params,params_dict]    
    else:
        preY = best_model.predict(X_test)
        preY = scaler_y.inverse_transform(preY)
        Y_pres = pd.DataFrame(preY,index=Y_test.index,columns=Y_test.columns)
        Y_pres_all = expand_data(Y_pres,colName='Lasso_forecasts')
        Y_test_all = expand_data(Y_test,colName='Load')
        Y_test_all = Y_test_all.merge(Y_pres_all,left_on='Date', right_on='Date', how='inner')
        return Y_test_all,best_params

def trainLEAR(res_dict,ifRecalibrate):
    """
    res_dict: results from XYtables, {'data':[X_train,X_test,Y_train,Y_test],
                				      'scalers':[scaler_x,scaler_y],
                				      'cols':[X_cols,Y_cols]}
    ifRecalibrate: True of False, indicator for price forecasting or not
    """
    # load data   
    X_train, X_test, Y_train, Y_test = res_dict['data']
    scaler_x,scaler_y = res_dict['scalers']
    X_train = scaler_x.inverse_transform(X_train)
    X_test = scaler_x.inverse_transform(X_test)
    Y_train = scaler_y.inverse_transform(Y_train)

    def find_columns(arr):
        columns_with_both = []
        for i in range(arr.shape[1]):
            unique_values = np.unique(arr[:, i])
            if len(unique_values) == 2 and 0 in unique_values and 1 in unique_values:
                columns_with_both.append(i)
        return columns_with_both
    
    columns_hol = find_columns(X_train)

    LEAR = MultiOutputLEAR()
    # preY = LEAR.recalibrate_predict(X_train, Y_train, X_test,columns_hol)

    if ifRecalibrate:
        Y_test_all,params_dict = rolling_train_price(X_train,Y_train,X_test,Y_test,best_model,scaler_x,scaler_y)
    else:
        preY = LEAR.recalibrate_predict(X_train, Y_train, X_test,columns_hol)
        # preY = scaler_y.inverse_transform(preY)
        
        Y_pres = pd.DataFrame(preY,index=Y_test.index,columns=Y_test.columns)
        Y_pres_all = expand_data(Y_pres,colName='Lasso_forecasts')
        Y_test_all = expand_data(Y_test,colName='Load')
        Y_test_all = Y_test_all.merge(Y_pres_all,left_on='Date', right_on='Date', how='inner')
        
    return Y_test_all,params_dict

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
    try:
        for train_index, test_index in tscv.split(X_train):
            print(train_index)
            X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
            y_train_fold, y_test_fold = Y_train.iloc[train_index], Y_train.iloc[test_index]
            
            model = MultiOutputRegressor(lgb.LGBMRegressor(**params))
            model.fit(X_train_fold,y_train_fold)
            preds = model.predict(X_test_fold, num_iteration=model.best_iteration_)
            mse = mean_squared_error(y_test_fold, preds)
            mse_scores.append(mse)
        average_mse = np.mean(mse_scores)
    except Exception as e:
        print(f"An error occurred: {e}")
        average_mse = float('inf')
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return average_mse

def trainLGB(res_dict,ifRecalibrate):
    """
    res_dict: results from XYtables, {'data':[X_train,X_test,Y_train,Y_test],
                				      'scalers':[scaler_x,scaler_y],
                				      'cols':[X_cols,Y_cols]}
    """
    X_train, X_test, Y_train, Y_test = res_dict['data']
    scaler_x,scaler_y = res_dict['scalers']

    study = optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=42),
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: lgbObjective(X_train,Y_train,trial), 
                   n_trials=100,
                   n_jobs=-1,
                   callbacks=[MaxTrialsCallback(n_trials=100, states=(TrialState.COMPLETE, TrialState.PRUNED))])

    best_params = study.best_params
    best_model = MultiOutputRegressor(lgb.LGBMRegressor(**best_params))
    best_model.fit(X_train, Y_train)
    if ifRecalibrate:
        Y_test_all = rolling_train_price(X_train,Y_train,X_test,Y_test,best_model,scaler_y)
    else:
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
        'activation': trial.suggest_categorical('activation', ['relu']),
        'solver': trial.suggest_categorical('solver', ['adam']),
        'alpha': trial.suggest_loguniform('alpha', 1e-5, 1e-2),
        'learning_rate_init': trial.suggest_loguniform('learning_rate_init', 1e-5, 1e-2),
        'max_iter': 200  # set higher iter times
    }

    # save mse for each fold
    mse_scores = []
    try:
        for train_index, test_index in tscv.split(X_train):
            X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
            y_train_fold, y_test_fold = Y_train.iloc[train_index], Y_train.iloc[test_index]
            
            model = MLPRegressor(**params, random_state=0, early_stopping=True, 
                         validation_fraction=0.1, n_iter_no_change=10)
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

def trainMLP(res_dict,ifRecalibrate):
    """
    res_dict: results from XYtables, {'data':[X_train,X_test,Y_train,Y_test],
                				      'scalers':[scaler_x,scaler_y],
                				      'cols':[X_cols,Y_cols]}
    """
    X_train, X_test, Y_train, Y_test = res_dict['data']
    scaler_x,scaler_y = res_dict['scalers']

    study = optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=42),
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: mlpObjective(X_train,Y_train,trial), 
                   n_trials=100,
                   n_jobs=-1,
                   callbacks=[MaxTrialsCallback(n_trials=100, states=(TrialState.COMPLETE, TrialState.PRUNED))])

    best_params = study.best_params
    best_model = MLPRegressor(**best_params,random_state=0, early_stopping=True, 
                         validation_fraction=0.1, n_iter_no_change=10)
    best_model.fit(X_train, Y_train)
    if ifRecalibrate:
        Y_test_all = rolling_train_price(X_train,Y_train,X_test,Y_test,best_model,scaler_y)
    else:
        preY = best_model.predict(X_test)
        preY = scaler_y.inverse_transform(preY)
        Y_pres = pd.DataFrame(preY,index=Y_test.index,columns=Y_test.columns)
        Y_pres_all = expand_data(Y_pres,colName='MLP_forecasts')
        Y_test_all = expand_data(Y_test,colName='Load')
        Y_test_all = Y_test_all.merge(Y_pres_all,left_on='Date', right_on='Date', how='inner')
    
    return Y_test_all, best_params

# Extratrees model with optuna
def ETRObjective(X_train, Y_train, trial):
    """
    trial: compute trial by optuna
    """
    tscv = TimeSeriesSplit(n_splits=5)

    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'random_state': 42
    }

    # Save MSE for each fold
    mse_scores = []
    try:
        for train_index, test_index in tscv.split(X_train):
            X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
            y_train_fold, y_test_fold = Y_train.iloc[train_index], Y_train.iloc[test_index]
            
            model = ExtraTreesRegressor(**params)
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

def trainETR(res_dict, ifRecalibrate):
    """
    res_dict: results from XYtables, {'data':[X_train,X_test,Y_train,Y_test],
                                      'scalers':[scaler_x,scaler_y],
                                      'cols':[X_cols,Y_cols]}
    """
    X_train, X_test, Y_train, Y_test = res_dict['data']
    scaler_x, scaler_y = res_dict['scalers']

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42),
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: ETRObjective(X_train, Y_train, trial), 
                   n_trials=100,
                   n_jobs=-1,
                   callbacks=[MaxTrialsCallback(n_trials=100, states=(TrialState.COMPLETE, TrialState.PRUNED))])

    best_params = study.best_params
    best_model = ExtraTreesRegressor(**best_params)
    best_model.fit(X_train, Y_train)
    
    if ifRecalibrate:
        Y_test_all = rolling_train_price(X_train, Y_train, X_test, Y_test, best_model, scaler_y)
    else:
        preY = best_model.predict(X_test)
        preY = scaler_y.inverse_transform(preY)
        Y_pres = pd.DataFrame(preY, index=Y_test.index, columns=Y_test.columns)
        Y_pres_all = expand_data(Y_pres, colName='ETR_forecasts')
        Y_test_all = expand_data(Y_test, colName='Load')
        Y_test_all = Y_test_all.merge(Y_pres_all, left_on='Date', right_on='Date', how='inner')
    
    return Y_test_all, best_params

# evaluation basic funcs
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

def evaluate_deterministic(Y):
    pred = Y.columns[1]
    obs = Y.columns[0]
    date_list = Y.index.tolist()
    Y['horizon'] = [dt.date() for dt in date_list]
    
    Y['error'] = Y[pred] - Y[obs]
    
    Y['error_square'] = Y['error']**2
    Y['mae_error'] = Y['error'].abs()
    # Y['mape_error'] = Y['mae_error']/np.abs(Y[obs])*100

    df_eval_det = pd.DataFrame(columns=['RMSE', 'MAE'])
    df_eval_det['RMSE'] = Y.groupby('horizon').mean()['error_square']**0.5
    # df_eval_det['MAPE'] = Y.groupby('horizon').mean()['mape_error']
    df_eval_det['MAE'] = Y.groupby('horizon').mean()['mae_error']

    return df_eval_det

def calculate_rmse_mae(Y):
    true_values = Y.iloc[:, 0]
    predicted_values = Y.iloc[:, 1]

    rmse = np.sqrt(np.mean((true_values - predicted_values)**2))

    mae = np.mean(np.abs(true_values - predicted_values))
    
    return rmse, mae

def eval_benchmarks(folder_path):
    location_names = []
    lasso_rmse, lasso_mae = [],[]
    lgb_rmse,lgb_mae = [],[]
    mlp_rmse,mlp_mae = [],[]

    pkl_files = glob.glob(os.path.join(folder_path, '*', 'NoText', 'lasso_lgb_mlp.pkl'))

    for pkl_path in pkl_files:
        location_name = os.path.basename(os.path.dirname(os.path.dirname(pkl_path)))
        location_names.append(location_name)
        print(pkl_path)
        with open(pkl_path, 'rb') as f:
            lasso_lgb_mlp = pickle.load(f)
        eval_lasso = evaluate_deterministic(lasso_lgb_mlp['lasso_fore']).mean()
        lasso_rmse.append(eval_lasso[0])
        lasso_mae.append(eval_lasso[1])

        eval_lgb = evaluate_deterministic(lasso_lgb_mlp['lgb_fore']).mean()
        lgb_rmse.append(eval_lgb[0])
        lgb_mae.append(eval_lgb[1])

        eval_mlp = evaluate_deterministic(lasso_lgb_mlp['mlp_fore']).mean()
        mlp_rmse.append(eval_mlp[0])
        mlp_mae.append(eval_mlp[1])

    eval_benchmark_df = pd.DataFrame()
    eval_benchmark_df['Locations'] = location_names
    eval_benchmark_df['lasso_rmse'], eval_benchmark_df['lasso_mae'] = lasso_rmse, lasso_mae
    eval_benchmark_df['lgb_rmse'], eval_benchmark_df['lgb_mae'] = lgb_rmse,lgb_mae
    eval_benchmark_df['mlp_rmse'], eval_benchmark_df['mlp_mae'] = mlp_rmse,mlp_mae
    return eval_benchmark_df

def eval_rte_benchmark():
    df = pd.read_csv('02-electricity-data/FranceNation/processed_FranceNation_load_J1.csv')
    # df = pd.read_csv('02-electricity-data/FranceNation/processed_FranceNation_load.csv')
    df = df.drop_duplicates(subset=['Date'])
    df = resampleDataFrame(df)
    df = df[df.index>='2023-01-02']
    eval_rte = evaluate_deterministic(df).mean()
    print(eval_rte)