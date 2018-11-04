import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
import os

from load_data import denormalize


def xgb_predict_all(X, Y, dir_prefix):
    """
    Predict all timestamps and features using xgb models.
    """
    result_list = []
    for i in range(Y[0].shape[0]):
        result_one_timestamp = []
        for j in range(Y[0].shape[1]):
            name = Y[0].columns[j]
            
            X_added = add_cycle_to_X(X, all_raw, name, i, 7)
            xg_reg = pickle.load(open('{}_{}_{}.model'.format(dir_prefix, name, i), 'rb'))
            result = pd.Series(xg_reg.predict(X_added))
    
            indexes = ['%d_%02d' % (station_id, i + 4) for station_id in range(90001, 90011)]
            result.index = indexes
            result_one_timestamp.append(result)
        result_list.append(pd.DataFrame(result_one_timestamp).T)
    result = pd.concat(result_list)
    result.columns = Y[0].columns
    result.index.name = 'FORE_data'
    result = denormalize(result)
    return result


def add_previous_to_X(X, prediction_list):
    if len(prediction_list) == 0:
        return X
    predictions = pd.concat(prediction_list, axis=1)
    predictions.columns = ['prediction_{}'.format(i) for i in range(predictions.shape[1])]
    return pd.concat([X, predictions], axis=1)


def construct_submission(raw_data, prediction, forecast_date, file_dir=None):
    """
    Construct submission data frame using original data and forecast result.
    """
    forecast_date = pd.Timestamp(forecast_date)
    origin_list = []
    for station_id in range(90001, 90011):
        origin_data = raw_data.loc[station_id, forecast_date][prediction.columns].iloc[:4]
        origin_data.index = ['%d_%02d' % (station_id, i) for i in range(4)]
        origin_list.append(origin_data)
    origin = pd.concat(origin_list)
    submission = pd.concat([origin, prediction])
    submission.columns = [name.split('_')[0] for name in submission.columns]
    submission.index.name = 'FORE_data'
    submission = submission.sort_index()

    if file_dir is None:
        file_dir = os.path.join('submission', 'forecast-{}.csv'.format(forecast_date.strftime('%Y%m%d%H')))
    with open(file_dir, 'w') as f:
        f.write('%10s,%10s,%10s,%10s' % ('FORE_data', 't2m', 'rh2m', 'w10m'))
        for index, value in submission.iterrows():
            f.write('\n%10s,%10.5f,%10.5f,%10.5f' % (index, value['t2m'], value['rh2m'], value['w10m']))
    return submission


def plot_prediction(y, prediction):
    for i in range(len(y)):
        plt.figure(figsize=(20, 3))
        for j in range(3):
            column = prediction.columns[j]
            plt.subplot(1, 3, j+1, ylabel=column)
            if j == 1:
                plt.title(range(90001, 90011)[i])
            plt.plot(range(33), prediction.sort_index().iloc[33*i:33*(i+1)][column], color='red', label='pre')
            plt.plot(range(33), denormalize(y[i])[column], color='green', label='true')
            plt.legend()


def calculate_rmse(y, prediction):
    result = []
    for i in range(len(y)):
        result.append([np.sqrt(mean_squared_error(denormalize(y[i])[column],
                                                  prediction.sort_index().iloc[33*i:33*(i+1)][column]))
                       for column in prediction.columns])
    return pd.DataFrame(result, index=range(90001, 90011), columns=prediction.columns)


def load_numpy_prediction(file_path):
    """
    Turn Deep Learning output numpy array into the same format as XGBoost prediction result.
    """
    pre_np = np.load(file_path)
    pre_np = pre_np.reshape([3, 330]).T
    indexes = pd.concat([pd.Series(['%d_%02d' % (station, seq) for seq in range(4, 37)]) for station in range(90001, 90011)])
    indexes.name = 'FORE_data'
    pre_df = pd.DataFrame(pre_np, columns=['t2m_obs', 'w10m_obs', 'rh2m_obs'], index=indexes)
    pre_df = pre_df[['t2m_obs', 'rh2m_obs', 'w10m_obs']]
    return pre_df


def eval_timespan(start, end, eval_set, dir_prefix, days=2):
    """
    Use evaluation set to generate a summarize of rmse.
    """
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    y_list = []
    pre_list = []
    for date in pd.date_range(start, end):
        x, y = generate_x_oneday(date, eval_set, days)
        y_list.append(y)
        pre = xgb_predict_all(x, y, dir_prefix)
        pre_list.append(pre)
    
    # Calculate RMSE for each hour
    hour_list = []
    columns = None
    for i in range(33):
        pre_hour = pd.concat([pre.iloc[(i*10):((i+1)*10)] for pre in pre_list])
        y_hour = denormalize(pd.concat([pd.DataFrame([y_one_station.iloc[i] for y_one_station in y]) for y in y_list]))
        columns = y_hour.columns
        row = []
        for column in columns:
            row.append(mse(pre_hour[column], y_hour[column]))
        hour_list.append(row)
    hour_rmse = pd.DataFrame(np.sqrt(hour_list), columns=columns)
    
    # Calculate RMSE for each station
    station_list = []
    for i in range(10):
        pre_station = pd.concat([pre.sort_index().iloc[(i*33):((i+1)*33)] for pre in pre_list])
        y_station = denormalize(pd.concat([y[i] for y in y_list]))
        row = []
        for column in columns:
            row.append(mse(pre_station[column], y_station[column]))
        station_list.append(row)
    station_rmse = pd.DataFrame(np.sqrt(station_list), columns=columns, index=range(90001, 90011))
    
    # Calculate overall RMSE for three features
    all_rmse = pd.DataFrame(np.sqrt(np.mean(station_list, axis=0)), index=columns)
    
    return hour_rmse, station_rmse, all_rmse
