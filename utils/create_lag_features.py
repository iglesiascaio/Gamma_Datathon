from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def create_lag_features(df, ts_col, lags):
   
    scaler = StandardScaler()
    
    #partial = pd.Series(data=pacf(y, nlags=48))
    #lags = list(partial[np.abs(partial) >= 0.2].index)
    
    df_out = df.copy()
    
    # avoid to insert the time series itself
    #lags.remove(0)
    feat_cols = []
    for l in lags:
        df_out[f"lag_{l}"] = df[ts_col].shift(l)
        feat_cols.append(f"lag_{l}")
    df_out[feat_cols] = scaler.fit_transform(df_out[feat_cols])

    
    return df_out