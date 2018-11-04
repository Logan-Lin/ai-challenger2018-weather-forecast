import xgboost as xgb
import pickle
import pandas as pd
import os

from matplotlib import pyplot as plt


params = {'objective': 'reg:linear', 'colsample_bytree': 0.3, 'learning_rate': 0.1,
          'max_depth': 5, 'alpha': 100, 'n_estimators': 100}


def fetch_labels(predict_index: int, column: str, y_list: list):
    """
    Fetch labels from y list.
    """
    return pd.Series([y.loc[predict_index + 4][column] for y in y_list])


def train_all_xgb(X, Y, X_va, Y_va, dir_prefix, params, start=0, end=None):
    """
    Train models for all time series and features.
    """
    if end is None:
        end = Y[0].shape[0]
    y_train_list = {column: [] for column in Y[0].columns}
    y_eval_list = {column: [] for column in Y[0].columns}
    for i in range(start, end):
        print('Training {}th day models'.format(i + 1))
        for j in range(len(Y[0].columns)):
            column = Y[0].columns[j]
            
            X_added = add_cycle_to_X(X, a_raw, column, i, 7)
            X_va_added = add_cycle_to_X(X_va, a_raw, column, i, 7)
            
            y_train = fetch_labels(i, column, Y)
            y_validation = fetch_labels(i, column, Y_va)
            
            max_depth = int(params.iloc[i*3+j]['max_depth']) 
            min_child_weight = int(params.iloc[i*3+j]['min_child_weight'])
            xg_reg = xgb.XGBRegressor(max_depth=max_depth, learning_rate=0.15, n_estimators=150, silent=True, 
                                      objective='reg:linear', booster='gbtree', n_jobs=47, alpha=10,
                                      gamma=0, min_child_weight=min_child_weight, max_delta_step=0, subsample=0.9, 
                                      colsample_bytree=0.9, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, 
                                      scale_pos_weight=1, base_score=0.5, random_state=0, seed=None, missing=None)
            xg_reg.fit(X_added, y_train, 
                       eval_set=[(X_va_added, y_validation)], verbose=False, 
                       eval_metric='rmse', early_stopping_rounds=15)
            
            # Save model to file.
            pickle.dump(xg_reg, open('{}_{}_{}.model'.format(dir_prefix, column, i), 'wb'))


def train_eval_xgb_model(X, Y, X_va, Y_va, params, column, predict_day):
    y_train = fetch_labels(predict_day, column, Y)
    y_eval = fetch_labels(predict_day, column, Y_va)
    xg_reg = xgb.XGBRegressor(params=params)
    print('Training {} {}th day model'.format(column, predict_day+1))
    xg_reg.fit(X, y_train, eval_set=[(X_va, y_eval)], verbose=False)
    
    prediction = xg_reg.predict(X_va)
    
    eval_df = pd.DataFrame(y_eval, columns=[column])
    eval_df = denormalize(eval_df)
    pre_df = pd.DataFrame(prediction, columns=[column])
    pre_df = denormalize(pre_df)
    eval_df.columns = eval_df.columns + '_eval'
    pre_df.columns = pre_df.columns + '_pre'
    compare_df = pd.concat([eval_df, pre_df], axis=1)
    
    plt.figure(figsize=(20, 10))
    for column in compare_df.columns:
        plt.plot(compare_df[column], label=column)
    plt.legend()
    
    return mean_squared_error(compare_df.iloc[:, 0], compare_df.iloc[:, 1])


def train_eval_xgb_model(X, Y, X_va, Y_va, column, predict_hour, output=False):
    """
    Evaluate one spot XGBoost model to test parameters.
    """
    y_train = fetch_labels(predict_hour, column, Y)
    y_eval = fetch_labels(predict_hour, column, Y_va)
    
    X_added = add_cycle_to_X(X, a_raw, column, predict_hour, 7)
    X_va_added = add_cycle_to_X(X_va, a_raw, column, predict_hour, 7)
    
    rmse_list = []
    for max_depth in range(3, 18, 1):
        for min_child_weight in range(0, 12, 1):
            xg_reg = xgb.XGBRegressor(max_depth=max_depth, learning_rate=0.15, n_estimators=150, silent=True, 
                                      objective='reg:linear', booster='gbtree', n_jobs=47, alpha=10,
                                      gamma=0, min_child_weight=min_child_weight, max_delta_step=0, subsample=0.9, 
                                      colsample_bytree=0.9, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, 
                                      scale_pos_weight=1, base_score=0.5, random_state=0, seed=None, missing=None)
            xg_reg.fit(X_added, y_train, 
                       eval_set=[(X_va_added, y_eval)], verbose=False, 
                       eval_metric='rmse', early_stopping_rounds=15)

            prediction = xg_reg.predict(X_va_added)

            eval_df = pd.DataFrame(y_eval, columns=[column])
            eval_df = denormalize(eval_df)
            pre_df = pd.DataFrame(prediction, columns=[column])
            pre_df = denormalize(pre_df)
            eval_df.columns = eval_df.columns + '_eval'
            pre_df.columns = pre_df.columns + '_pre'
            compare_df = pd.concat([eval_df, pre_df], axis=1)

            rmse = sqrt(mean_squared_error(compare_df.iloc[:, 0], compare_df.iloc[:, 1]))
            rmse_list.append([max_depth, min_child_weight, rmse])
            if output:
                print('{}-{}-{}'.format(max_depth, min_child_weight, rmse))
    return pd.DataFrame(rmse_list, columns=['max_depth', 'min_child_weight', 'rmse'])


def gen_best_parameters(X, Y, X_va, Y_va):
    result_list = []
    index_list = []
    for day in range(0, 33):
        print('Testing {}th day best parameters'.format(day))
        for name in Y[0].columns:
            result = train_eval_xgb_model(X, Y, X_va, Y_va, name, day)
            result = result.sort_values(by='rmse').iloc[0]
            index_list.append((day, name))
            result_list.append(result)
    best_parameters = pd.DataFrame(result_list, index=index_list)
    return best_parameters

