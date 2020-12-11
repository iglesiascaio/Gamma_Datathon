from dicts import periods_dict
import pandas as pd

def add_periods(df):
    out_df = df.copy()
    for col in periods_dict.keys():
        out_df[col] = pd.Series([])
        for start, end in periods_dict[col].keys():
            out_df.loc[start:end, col] = periods_dict[col][(start,end)]
        out_df[col] = out_df[col].fillna(0)

    return out_df 