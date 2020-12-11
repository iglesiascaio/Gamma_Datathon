import pandas as pd

def read_csvs(years):
    df_convention = pd.DataFrame()
    df_washington = pd.DataFrame()
    df_sts = pd.DataFrame()
    for year in years:
        append_convention = pd.read_csv('./clean_opendata/' + year + '/convention.csv')
        df_convention = df_convention.append(append_convention)
        append_washington = pd.read_csv('./clean_opendata/' + year + '/washington.csv')
        df_washington = df_washington.append(append_washington)
        append_sts = pd.read_csv('./clean_opendata/' + year + '/sts.csv')
        df_sts = df_sts.append(append_sts)
        
    return df_convention, df_washington, df_sts