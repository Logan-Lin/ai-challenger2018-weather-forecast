from netCDF4 import Dataset
import os
import pandas as pd
import numpy as np


explain = pd.read_csv(os.path.join('data', 'explain.csv'), index_col=0).dropna(how='any', axis=1)
train_dir = os.path.join('data', 'ai_challenger_wf2018_trainingset_20150301-20180531.nc')
validation_dir = os.path.join('data', 'ai_challenger_wf2018_validation_20180601-20180828_20180905.nc')


def load_raw_file(file_dir: str) -> pd.DataFrame:
    """
    Load data from original netCDF4 file.

    :return: one pd.DataFrame containing all data in file.
    """
    print("Loading netCDF4 file", file_dir)
    data = Dataset(file_dir)
    date_list = [pd.Timestamp.strptime(str(int(this_date)), '%Y%m%d%H') for this_date in data['date']]
    station_list = list(data['station'][:])
    feature_list = list(data.variables.keys())[3:]

    date_index = []
    for date in date_list:
        date_index += [date] * data['foretimes'][:].shape[0]
    foretime_index = list(data['foretimes'][:]) * data['date'][:].shape[0]

    df_list = []
    for i in range(len(station_list)):
        print("Loading station", station_list[i])
        df = pd.DataFrame()
        df['date'] = date_index
        df['foretime'] = foretime_index
        for feature in feature_list:
            feature_df = pd.DataFrame(data[feature][:, :, i].reshape(-1, 1), columns=[feature])
            df = pd.concat([df, feature_df], axis=1)
        df['station_id'] = station_list[i]
        df_list.append(df)
    result = pd.concat(df_list)
    result = result.set_index(['station_id', 'date', 'foretime'])
    return result


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize data.
    """
    result = pd.DataFrame(df, copy=True)
    for name in df.columns:
        scope = explain.loc[name]['scope'][1:-1].split(',')
        min_value = float(scope[0].strip())
        max_value = float(scope[1].strip())

        result[name] = (result[name] - min_value) / (max_value - min_value)
    return result


def denormalize(df: pd.DataFrame):
    """
    Retrieve normalized data into real value.
    """
    result = pd.DataFrame(df, copy=True)
    for name in df.columns:
        scope = explain.loc[name]['scope'][1:-1].split(',')
        min_value = float(scope[0].strip())
        max_value = float(scope[1].strip())

        result[name] = result[name] * (max_value - min_value) + min_value
    return result


def check_empty_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a pandas DataFrame with all dates with empty data.
    """
    result = set()
    for name in df.columns:
        empty_index_set = set(df.loc[df[name].apply(np.isnan)].groupby(['station_id', 'date']).groups.keys())
        result |= empty_index_set
    result = pd.DataFrame(list(result), columns=['station_id', 'date'])
    result = result.sort_values(by=['station_id', 'date'])
    result = result.set_index('station_id')
    return result


def check_invalid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check all values that exceeds valid scope and turn them into np.nan.
    """
    result = pd.DataFrame(df, copy=True)
    for name in df.columns:
        scope = explain.loc[name]['scope'][1:-1].split(',')
        min_value = float(scope[0].strip())
        max_value = float(scope[1].strip())

        result.loc[result[name] < min_value, name] = np.nan
        result.loc[result[name] > max_value, name] = np.nan
    return result
