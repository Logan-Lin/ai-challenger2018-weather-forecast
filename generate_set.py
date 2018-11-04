import pandas as pd
import numpy as np
from load_data import normalize, check_invalid

obs_names = ['psur_obs', 't2m_obs', 'q2m_obs', 'w10m_obs', 'd10m_obs', 'rh2m_obs', 'u10m_obs', 'v10m_obs', 'RAIN_obs']
m_names = ['psfc_M', 't2m_M', 'q2m_M', 'rh2m_M', 'w10m_M', 'd10m_M', 'u10m_M', 'v10m_M', 'SWD_M', 'GLW_M', 'HFX_M',
           'LH_M', 'RAIN_M', 'PBLH_M', 'TC975_M', 'TC925_M', 'TC850_M', 'TC700_M', 'TC500_M', 'wspd975_M', 'wspd925_M',
           'wspd850_M', 'wspd700_M', 'wspd500_M', 'Q975_M', 'Q925_M', 'Q850_M', 'Q700_M', 'Q500_M']

holiday = ['2017-01-01', '2017-01-02', '2017-01-27', '2017-01-28', '2017-01-29',
           '2017-01-30', '2017-01-31', '2017-02-01', '2017-02-02', '2017-04-02',
           '2017-04-03', '2017-04-04', '2017-04-29', '2017-04-30', '2017-05-01',
           '2017-05-28', '2017-05-29', '2017-05-30', '2017-10-01', '2017-10-02',
           '2017-10-03', '2017-10-04', '2017-10-05', '2017-10-06', '2017-10-07',
           '2017-10-08', '2017-12-30', '2017-12-31',
           '2018-01-01', '2018-02-15', '2018-02-16', '2018-02-17', '2018-02-18',
           '2018-02-19', '2018-02-20', '2018-02-21', '2018-04-05', '2018-04-06',
           '2018-04-07', '2018-04-29', '2018-04-30', '2018-05-01', '2018-06-16',
           '2018-06-17', '2018-06-18']
work = ['2017-01-22', '2017-02-04', '2017-04-01', '2017-05-27', '2017-09-30',
        '2018-02-11', '2018-02-24', '2018-04-08', '2018-04-28']

def fill_nan_with_m(df: pd.DataFrame):
    """
    Use M data to fill all the NaNs in observation data.
    """
    obs_columns = ['psur_obs', 't2m_obs', 'q2m_obs', 'rh2m_obs', 'w10m_obs', 
                   'd10m_obs', 'u10m_obs', 'v10m_obs', 'RAIN_obs']
    m_columns = ['psfc_M', 't2m_M', 'q2m_M', 'rh2m_M', 'w10m_M', 'd10m_M',
                'u10m_M', 'v10m_M', 'RAIN_M']
    result = pd.DataFrame(df, copy=True)
    for (obs_column, m_column) in zip(obs_columns, m_columns):
        result[obs_column].fillna(result[m_column], inplace=True)
    return result


def generate_x(station_id: int, forecast_date: pd.Timestamp, df, days=2, predict=False):
    history_obs, history_m, prediction = generate_one_set(station_id, forecast_date, df, days, predict)
    obs_s = generate_history_obs_data(history_obs)
    m_s = generate_history_m_data(history_m)
    date_f = generate_date_feature(forecast_date)
    week_s_list = []
    for i in range(4):
        week_s_list.append(generate_stat_feature(station_id, forecast_date - pd.Timedelta(7 * i, 'D'), df, 7, 'week_{}_s'.format(i + 1)))
    month_s = generate_stat_feature(station_id, forecast_date, df, 30, 'month_s')
    station_onehot = get_onehot(station_id, 90001, 90010, name='station')

    return pd.concat([obs_s, m_s, date_f] + week_s_list + [month_s, station_onehot]), prediction


def generate_x_oneday(forecast_date, df: pd.DataFrame, days=2):
    forecast_date = pd.Timestamp(forecast_date)
    x_list = []
    y_list = []
    for station in range(90001, 90011):
        x, y = generate_x(station, forecast_date, df, days=days, predict=True)
        x_list.append(x)
        y_list.append(y)
    return pd.DataFrame(x_list), y_list


def generate_one_set(station_id: int, forecast_date: pd.Timestamp, df: pd.DataFrame, previous_days=2, predict=False):
    """
    Use forecast date's data and previous date data to concat a set of training data.
    """
    forecast_date = pd.Timestamp(forecast_date)
    history_list = []
    for i in range(previous_days, 0, -1):
        history_list.append(df.loc[station_id, forecast_date - pd.DateOffset(days=i)].iloc[:24])
    history = pd.concat(history_list + [df.loc[station_id, forecast_date]])
    history = check_invalid(history)
    history = history.interpolate(method='linear', limit=8, limit_direction='both')
    # history = fill_nan_with_m(history)
    if predict:
        assert ~history.iloc[:(24 * previous_days + 4)].isnull().values.any(), 'Empty data found in station {} date {}'.format(station_id, forecast_date)
    else:
        assert ~history.isnull().values.any(), 'Empty data found in station {} date {}'.format(station_id, forecast_date)
    history = normalize(history)

    history_obs = history.iloc[:(24 * previous_days + 4)][obs_names]
    history_m = history[m_names]

    prediction = history.iloc[(24 * previous_days + 4):][['t2m_obs', 'rh2m_obs', 'w10m_obs']]

    return history_obs, history_m, prediction


def get_onehot(value, min_value, max_value, step=1, name="") -> pd.Series:
    length = int((max_value - min_value) / step) + 1
    result = np.zeros([length])
    index = int((value - min_value) / step)
    result[index] = 1

    name_list = [name + "_{}".format(i) for i in range(length)]
    result = pd.Series(result, index=name_list)
    return result


def generate_series_data(df: pd.DataFrame, column_length: int) -> pd.Series:
    df_max = df.max()
    df_max.index = df_max.index + '_max'
    df_min = df.min()
    df_min.index = df_min.index + '_min'
    df_mean = df.mean()
    df_mean.index = df_mean.index + '_mean'
    df_var = df.var()
    df_var.index = df_var.index + '_var'

    df_series = df.stack()
    series_index = np.array([[i] * column_length for i in range(int(len(df_series.index) / column_length))]).reshape(1, -1)[0].astype(str)
    df_series.index = df_series.index.get_level_values(1) + '_' + series_index

    return pd.concat([df_series, df_max, df_min, df_mean, df_var])


def generate_history_m_data(m_df: pd.DataFrame) -> pd.Series:
    """
    Use Ruitu history data frame generated by function generate_one_set()
    to construct training data.
    """
    return generate_series_data(m_df, len(m_names))


def generate_history_obs_data(obs_df: pd.DataFrame) -> pd.Series:
    """
    Use observation history data frame generated by function generate_one_set()
    to construct training data.
    """
    return generate_series_data(obs_df, len(obs_names))


def generate_date_feature(date: pd.Timestamp):
    """
    Generate date features.
    """
    result = pd.Series()
    result['timestamp'] = date.timestamp()

    format_string = '%Y-%m-%d'
    dt_string = date.strftime(format_string)
    result['holiday'] = int((dt_string in holiday) or (date.weekday() in [5, 6] and dt_string not in work))

    result = pd.concat([result, get_onehot(date.weekday(), 0, 6, name='weekday')])
    return result


def generate_stat_feature(station_id: int, forecast_date: pd.Timestamp, df: pd.DataFrame, days, name):
    """
    Using assigned length of history data to fetch statistic features.
    """
    history_list = []
    for i in range(days, 0, -1):
        try:
            history_list.append(df.loc[station_id, forecast_date - pd.DateOffset(days=i)].iloc[:24])
        except KeyError:
            pass
    h = normalize(pd.concat(history_list + [df.loc[station_id, forecast_date]]))

    h_max = h.max()
    h_min = h.min()
    h_mean = h.mean()
    h_var = h.var()

    h_max.index = h_max.index + '_{}_max'.format(name)
    h_min.index = h_min.index + '_{}_min'.format(name)
    h_mean.index = h_mean.index + '_{}_mean'.format(name)
    h_var.index = h_var.index + '_{}_var'.format(name)

    return pd.concat([h_max, h_min, h_mean, h_var])


def generate_period(begin, end, df: pd.DataFrame, days=2):
    """
    Generate training data using a specified period.
    """
    return combine_periods([(begin, end)], df, days)


def combine_periods(periods: list, df: pd.DataFrame, days=2):
    """
    Combine training data fetched from multiple periods.
    """
    x_list = []
    y_list = []
    date_list = []
    date_index = None
    for period in periods:
        if date_index is None:
            date_index = pd.date_range(period[0], period[1], freq='D')
        else:
            date_index = date_index.append(pd.date_range(period[0], period[1], freq='D'))
    for date in date_index:
        for station in range(90001, 90011):
            try:
                x, y = generate_x(station, date, df, days=days)
                x_list.append(x)
                y_list.append(y)
                date_list.append(date)
            except AssertionError:
                pass
    return pd.DataFrame(x_list), y_list, date_list


def fetch_labels(predict_index: int, column: str, y_list: list):
    """
    Fetch labels from y list.
    """
    return pd.Series([y.loc[predict_index + 4][column] for y in y_list])


def add_cycle_to_X(X, raw, column, predict_hour, add_days, interval=1):
    explain = pd.read_csv(os.path.join('data', 'explain.csv'), index_col=0).dropna(how='any', axis=1)
    
    def get_station_id(X):
        station_list = []
        for index, row in X.iterrows():
            station_onehot = row[['station_{}'.format(i) for i in range(10)]]
            station_id = int('900%02d' % int(np.sum([station_onehot[i] * (i+1) for i in range(10)])))
            station_list.append(station_id)
        return station_list

    def normalize_single_column(df: pd.DataFrame, name) -> pd.DataFrame:
        """
        Normalize data.
        """
        result = pd.DataFrame(df, copy=True)
        scope = explain.loc[name]['scope'][1:-1].split(',')
        min_value = float(scope[0].strip())
        max_value = float(scope[1].strip())

        result[name] = (result[name] - min_value) / (max_value - min_value)
        return result
    
    origin_dates = [pd.Timestamp.fromtimestamp(timestamp) - pd.Timedelta('8 hours') for timestamp in X['timestamp']]
    station_list = get_station_id(X)
    series_list = []
    for day in range(add_days):
        day = (day + 1) * interval
        fetch_dates = pd.Series(origin_dates) - pd.Timedelta('{} days'.format(day))
        series = pd.Series([raw.loc[(station_id, date, predict_hour + 4)][column] 
                            for (station_id, date) in zip(station_list, fetch_dates)])
        series.name = column
        series_list.append(series)
    result = normalize_single_column(pd.DataFrame(series_list).T, column)
    result.columns = ['{}_{}*{}days'.format(column, interval, i + 1) for i in range(add_days)]
    return pd.concat([X, result], axis=1)
