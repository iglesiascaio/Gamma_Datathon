from utils.periods import add_periods
import pandas as pd

def create_reg_forecast_horizon(cols, dt_start, dt_end):
    X = pd.DataFrame()
    out_df = pd.DataFrame(columns=cols)
    dt_range = pd.date_range(dt_start, dt_end, freq='H')
    X['dt'] = dt_range
    X['year'] = X['dt'].dt.year.astype('category')
    X['month'] = X['dt'].dt.month.astype('category')
    X['week_day'] = X['dt'].dt.weekday.astype('category')
    X['time'] = X['dt'].dt.time.astype(str).apply(lambda x: x[:2]).astype('category')
    X.sort_values('dt', inplace = True)
    X = X.set_index('dt')
    X = add_periods(X)
    X.reset_index(inplace = True)
    X_dummies = pd.get_dummies(X)
    X = X_dummies.drop(columns = ['dt'])
    for col in X.columns:
        out_df[col] = X[col]
    out_df = out_df.fillna(0).values
    return out_df, dt_range