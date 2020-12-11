#!/usr/bin/env python3
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import calendar
import xgboost as xgb
from sklearn.metrics import mean_squared_error as MSE
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score as R2
from tqdm import tqdm

# Homemade functions
from utils.create_lag_features import create_lag_features
from utils.periods import add_periods

def plot_calculate_errors(model_name, model, n, train_data, test_data, test_dates, plot=False):
    if plot:
        plt.figure( figsize = (25,15))
    
    relative_errors = []
    errors = []
    r2_list = []
    errors = []
    models = []
    for i, (X_train, y_train) in tqdm(enumerate(train_data)):
        search = model.fit(X_train, y_train)
        models.append(model)
        
        if model_name == 'xgboost':
            print(" [.] Params:", search.steps[1][1].best_params_)
        
        y_pred = model.predict(test_data[i][0])
        error = MSE(test_data[i][1], y_pred)**(1/2)
        r2 = R2(test_data[i][1], y_pred)
        
        if plot:
            plt.subplot(n//2,2,i+1)
    
            # plot test and prediction values 
            plt.plot(test_dates[i], y_pred, label = 'prediction')
            plt.plot(test_dates[i], test_data[i][1], label = 'test', alpha = 0.5)
        
            # [Optional] Some extra details
            #  diff = y_pred_xg.reshape(test_data[i][1].shape) - test_data[i][1]
            #  plt.plot(test_dates[i],(diff**2)**(1/2))
            
            plt.ylabel('Débit horaire')
            plt.title('Débit horaire (prévision x real data)')
            plt.xticks(rotation = 45)
            plt.legend()
            
        relative_error = error/df_analysis[target].mean()
        errors.append(error)
        relative_errors.append(relative_error)
        r2_list.append(r2)
    
    plt.show()

    print(" [+] Model Name:", model_name)
    print(' [.] RMSE: {}'.format(str(errors)))
    print(' [.] RMSE mean: {}'.format(float(np.mean(errors))))
    print(' [.] Relative RMSE: {}'.format(str(relative_errors)))
    print(' [.] Relative RMSE Mean: {:.2%}'.format(float(np.mean(relative_errors))))
    print(' [.] R^2: {}'.format(str(r2_list)))
    print(' [.] R^2 Mean: {:.2%}'.format(float(np.mean(r2_list))))
    print("-----------------------------------------------------")
    
    rmse_mean[model_name] = float(np.mean(errors))
    relative_rmse_mean[model_name] = float(np.mean(relative_errors))
    r2_mean[model_name] = float(np.mean(r2_list))

    return models


def plot_calculate_errors_weekday(model_name, model, train, test, plot=False, target='Débit horaire'):
    if plot:
        plt.figure(figsize = (25,25))
            
    relative_errors = []
    errors = []
    r2_list = []
    errors = []
    for i in range(7):
        weekday_y_train = train_weekdays[i][['Débit horaire']].values
        weekday_dates_train = train_weekdays[i]['Date et heure de comptage']
        weekday_X_train = train_weekdays[i].drop(columns = ['Débit horaire', 'Taux d\'occupation', 'Date et heure de comptage', 'index']).values

        weekday_y_test = test_weekdays[i][['Débit horaire']].values
        weekday_dates_test = test_weekdays[i]['Date et heure de comptage']
        weekday_X_test = test_weekdays[i].drop(columns = ['Débit horaire', 'Taux d\'occupation', 'Date et heure de comptage', 'index']).values

        search = model.fit(weekday_X_train, weekday_y_train)

        y_pred = model.predict(weekday_X_test)
        error = MSE(weekday_y_test, y_pred)**(1/2)
        r2 = R2(weekday_y_test, y_pred)
        
        if model_name == 'xgbrabo weekday':
            print(" [.] Params:", search.steps[1][1].best_params_)
            
        if plot:
            plt.subplot(4,2,i+1)

            # plot test and prediction values 
            plt.plot(weekday_dates_test, y_pred, label='prediction')
            plt.plot(weekday_dates_test, weekday_y_test, label='test', alpha=0.5)

            # [Optional] Some extra details
            #  diff = y_pred_xg.reshape(list_test_data[i][1].shape) - list_test_data[i][1]
            #  plt.plot(test_dates[i],(diff**2)**(1/2))

            plt.ylabel('Débit horaire')
            plt.title('Débit horaire (prévision x real data) for ' + calendar.day_name[i])
            plt.xticks(rotation = 45)
            plt.legend()


        # TODO: calculate the mean of every weekday
        #relative_error = error/df_analysis[target].mean()
        errors.append(error)
        #relative_errors.append(relative_error)
        r2_list.append(r2)
     
    plt.show()

    print(" [+] Model Name:", model_name)
    print(' [.] RMSE: {}'.format(str(errors)))
    print(' [.] RMSE mean: {}'.format(float(np.mean(errors))))
    #print(' [.] Relative RMSE: {}'.format(str(relative_errors)))
    #print(' [.] Relative RMSE Mean: {:.2%}'.format(float(np.mean(relative_errors))))
    print(' [.] R^2: {}'.format(str(r2_list)))
    print(' [.] R^2 Mean: {:.2%}'.format(float(np.mean(r2_list))))
    print("-----------------------------------------------------")

    rrmse_mean[model_name] = float(np.mean(errors))
    #relative_rmse_mean[model_name] = float(np.mean(relative_errors))
    r2_mean[model_name] = float(np.mean(r2_list))


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


def separate_last_n_weeks(n, df_analysis, X, y):
    test_dates = []
    train_data = []
    test_data = []
    for i in range(n, 0, -1):
        train_data.append((X[:-7*i*24], y[:-7*i*24]))
        test_data.append((X[-7*i*24:][:7*24], y[-7*i*24:][:7*24]))
        test_dates.append(df_analysis['Date et heure de comptage'].iloc[-7*i*24:].iloc[:7*24])

    return train_data, test_data, test_dates


start_date_knn = '2016-01-01'
end_date_knn = '2020-11-29'

start_date_xgboost = '2020-05-12'
end_date_xgboost = '2020-11-29'

dataset = "washington"      # {"sts", "convention", "washington"}
target = "Débit horaire"    # {"Débit horaire", "Taux d'occupation"}
n = 4                       # number of weeks to test

print("**********************************")
print("             Metadata")
print("**********************************")

print(" [.] Start date KNN:", start_date_knn)
print(" [.] End date KNN:", end_date_knn)
print(" [.] Start date xgboost:", start_date_xgboost)
print(" [.] End date xgboost:", end_date_xgboost)
print(" [.] Dataset:", dataset)
print(" [.] Target:", target)
print(" [.] N. tests:", n)

print("")
print("")

rmse_mean = dict()
relative_rmse_mean = dict()
r2_mean = dict()

years = ['2016', '2017','2018','2019', '2020']

df_convention, df_washington, df_sts = read_csvs(years)

df_convention['Date et heure de comptage'] = pd.to_datetime(df_convention['Date et heure de comptage'])
df_washington['Date et heure de comptage'] = pd.to_datetime(df_washington['Date et heure de comptage'])
df_sts['Date et heure de comptage'] = pd.to_datetime(df_sts['Date et heure de comptage'])

if dataset == 'convention':
    df_analysis = df_convention.copy()
elif dataset == 'sts':
    df_analysis = df_sts.copy()
elif dataset == 'washington':
    df_analysis = df_washington.copy()
else:
    print(" [Error] Invalid dataset")
    exit(1)

df_analysis = df_analysis[['Date et heure de comptage', target]]
df_analysis['year'] = df_analysis['Date et heure de comptage'].dt.year.astype('category')
df_analysis['month'] = df_analysis['Date et heure de comptage'].dt.month.astype('category')
df_analysis['week_day'] = df_analysis['Date et heure de comptage'].dt.weekday.astype('category')
df_analysis['time'] = df_analysis['Date et heure de comptage'].dt.time.astype(str).apply(lambda x: x[:2]).astype('category')
df_analysis.sort_values('Date et heure de comptage', inplace=True)
df_analysis.reset_index(inplace = True)

df_analysis.set_index('Date et heure de comptage', inplace=True)

# filter dates
df_analysis_knn = df_analysis[start_date_knn:end_date_knn]
df_analysis_xgboost = df_analysis[start_date_knn:end_date_xgboost]

add_periods(df_analysis_knn)
add_periods(df_analysis_xgboost)

df_analysis_knn.reset_index(inplace = True)
df_analysis_xgboost.reset_index(inplace = True)

# Add lag features
# df_analysis_knn = create_lag_features(df_analysis_knn, target, [6*24])
# df_analysis_xgboost = create_lag_features(df_analysis_xgboost, target, [6*24])

df_analysis_knn_dummies = pd.get_dummies(df_analysis_knn).dropna()
df_analysis_xgboost_dummies = pd.get_dummies(df_analysis_xgboost).dropna()

y_knn = df_analysis_knn_dummies[[target]].values
y_xgboost = df_analysis_xgboost_dummies[[target]].values

dates_knn = df_analysis_knn_dummies['Date et heure de comptage']
dates_xgboost = df_analysis_xgboost_dummies['Date et heure de comptage']

X_knn = df_analysis_knn_dummies.drop(columns = ['Date et heure de comptage', target, 'index']).values
X_xgboost = df_analysis_xgboost_dummies.drop(columns = ['Date et heure de comptage', target, 'index']).values

train_data_knn, test_data_knn, test_dates_knn = separate_last_n_weeks(n, df_analysis_knn, X_knn, y_knn)
train_data_xgboost, test_data_xgboost, test_dates_xgboost = separate_last_n_weeks(n, df_analysis_xgboost, X_xgboost, y_xgboost)

min_max_scaler = MinMaxScaler()

knn_regressor = KNeighborsRegressor(n_neighbors = 1)
pl_knn = make_pipeline(min_max_scaler, knn_regressor)

KNNs = plot_calculate_errors("KNN", pl_knn, n, train_data_knn, test_data_knn, test_dates_knn)

xgboost = xgb.XGBRegressor(
        eval_metric = 'rmse',
        nthread = 4,
        eta = 0.1,
        max_depth = 5,
        subsample = 0.5,
        colsample_bytree = 1.0,
        )

parameters = {
     'max_depth': [2, 3, 4, 5, 6, 10],
     'learning_rate': [0.1, 0.2, 0.3, 0.4],
     'subsample': [0.5, 0.6, 0.7, 0.8],
     'reg_lambda': [1.0, 5.0, 10.0],
     'n_estimators': [100, 200]
}

xgboost_search = RandomizedSearchCV(xgboost, parameters, n_jobs=1, cv=5)
pl_xgboost = make_pipeline(min_max_scaler, xgboost_search)

xgboosts = plot_calculate_errors("xgboost", pl_xgboost, n, train_data_xgboost, test_data_xgboost, test_dates_xgboost)

voting_regressor = VotingRegressor([('knn', pl_knn), ('xgboost', pl_xgboost)])
knns_xgboosts = plot_calculate_errors("xgboost + KNN average ensemble", voting_regressor, n, train_data_xgboost, test_data_xgboost, test_dates_xgboost)

# ## Ensemble Analysis

from mlxtend.regressor import StackingRegressor
from sklearn.svm import SVR
import warnings

regressors = [pl_xgboost, pl_knn]
svr_rbf = SVR(kernel='rbf')
stregr = StackingRegressor(regressors=regressors, 
                           meta_regressor=svr_rbf)

params = {'lasso__alpha': [0.1, 1.0, 10.0],
          'ridge__alpha': [0.1, 1.0, 10.0],
          'svr__C': [0.1, 1.0, 10.0],
          'meta_regressor__C': [0.1, 1.0, 10.0, 100.0],
          'meta_regressor__gamma': [0.1, 1.0, 10.0]}

grid = RandomizedSearchCV(stregr, 
                    params, 
                    cv=5,
                    refit=True)

ensembles = plot_calculate_errors("xgboost + knn stacking regressor", grid, n, train_data_xgboost, test_data_xgboost, test_dates_xgboost, plot=True)

df_weekday = []
for i in range(7):
    df_weekday.append(df_analysis_dummies[df_analysis_dummies[f'week_day_{i}'] == True])

#Making every day start at midnight
df_weekday[0].drop(df_weekday[0].tail(1).index, inplace=True) 
 
train_weekdays = []
test_weekdays = []

for i in range(7):
    train_weekdays.append(df_weekday[i].iloc[:-24])
    test_weekdays.append(df_weekday[i].iloc[-24:])

xgbrabo = xgb.XGBRegressor(
         base_score=0.5,
         booster='gbtree',
         importance_type='gain', 
         missing=None, 
         n_estimators=100, 
         nthread=7, 
         reg_alpha=0, 
         reg_lambda=1, 
         scale_pos_weight=1, 
         seed=0, 
         eval_metric='rmse')

parameters = {
    'eta': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    'n_estimators': [1000],
    'booster': ['gbtree', 'dart'],
    'reg_lambda': [1, 10]
}

xgboost_search = RandomizedSearchCV(xgbrabo, parameters, n_jobs=1, cv=5)
pl_xgboost = make_pipeline(min_max_scaler, xgboost_search)

plot_calculate_errors_weekday("xgbrabo weekday", pl_xgboost, train_weekdays, test_weekdays)

df_rmse_mean = pd.DataFrame.from_dict(rmse_mean, orient='index')
df_rmse_mean.columns = ['RMSE mean']
df_rmse_mean.sort_values(by=['RMSE mean'], inplace=True, ascending=True)
print(df_rmse_mean)

df_rel_rmse_mean = pd.DataFrame.from_dict(relative_rmse_mean, orient='index')
df_rel_rmse_mean.columns = ['Relative RMSE mean']
df_rel_rmse_mean.sort_values(by=['Relative RMSE mean'], inplace=True, ascending=True)
print(df_rel_rmse_mean)

df_r2_mean = pd.DataFrame.from_dict(r2_mean, orient='index')
df_r2_mean.columns = ['R^2 mean']
df_r2_mean.sort_values(by=['R^2 mean'], inplace=True, ascending=False)
print(df_r2_mean)


