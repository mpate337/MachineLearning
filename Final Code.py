# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 11:24:46 2019

@author: harsh
"""

#==============================================================================
#                           Importing Libraries
#==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime 
from sklearn.model_selection import KFold
import time
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import time 

#==============================================================================
#                  Data description and Data Preprocessing
#==============================================================================

#Loading train and test Datasets
train_df = pd.read_csv("train.csv", parse_dates=["first_active_month"])
test_df = pd.read_csv("test.csv", parse_dates=["first_active_month"])
print("Shape of train_df dataset : ",train_df.shape)
print("Shape of test_df dataset : ",test_df.shape)

#Analysis of Target feature of training set
train_df.target.describe()

plt.figure(figsize=(10,8))
plt.scatter(range(train_df.shape[0]), np.sort(train_df["target"].values))
plt.xlabel('Number of observations', fontsize=14)
plt.ylabel('Target value', fontsize=14)
plt.title("Distribution of target", fontsize=14)
plt.show()

plt.figure(figsize=(12, 5))
plt.hist(train_df.target.values, bins=150)
plt.ylabel('Total number', fontsize=14)
plt.xlabel('Target value', fontsize=14)
plt.title("Histogram Plot of Target value")
plt.show()

#sns.set_style("whitegrid")
#sns.violinplot(x=train_df.target.values)
#plt.show()

#Number of Outliers
train_target_unique = train_df.target.unique()
outliers_count=train_df[np.round(train_df["target"])<=-30.0].target.count()
print("Total number of outliers: ",outliers_count)
#2207 outliers

#To handle Outliers
train_df['outliers']=0
train_df.loc[train_df['target'] < -30, 'outliers'] = 1
train_df.loc[train_df['target'] <= -30.0, 'target'] = train_df.target.mean()

train_df.target.describe()

#Analysis of First Active Month and count for Training dataset
train_firstactivemonth_unique = train_df.first_active_month.dt.date.unique()

FirstActiveMonth_count = train_df.first_active_month.dt.date.value_counts()
FirstActiveMonth_count = FirstActiveMonth_count.sort_index()

plt.figure(figsize=(14,6))
sns.barplot(FirstActiveMonth_count.index, FirstActiveMonth_count.values, alpha=0.8, color='blue')
plt.xticks(rotation='vertical')
plt.ylabel('Number of cards', fontsize=16)
plt.xlabel('First active month', fontsize=16)
plt.title("First active month count in train set", fontsize=16)
plt.show()

#Analysis of First Active Month and count for Test dataset
test_firstactivemonth_unique = test_df.first_active_month.dt.date.unique()

FirstActiveMonth_count = test_df.first_active_month.dt.date.value_counts()
FirstActiveMonth_count = FirstActiveMonth_count.sort_index()

plt.figure(figsize=(14,6))
sns.barplot(FirstActiveMonth_count.index, FirstActiveMonth_count.values, alpha=0.8, color='blue')
plt.xticks(rotation='vertical')
plt.xlabel('First active month', fontsize=16)
plt.ylabel('Number of cards', fontsize=16)
plt.title("First active month count in test set", fontsize=16)
plt.show()

#unique values of features
train_df.feature_1.unique()
train_df.feature_2.unique()
train_df.feature_3.unique()

#Analysis of features
# feature 1
plt.figure(figsize=(8,4))
sns.violinplot(x="feature_1", y='target', data=train_df)
plt.xticks(rotation='vertical')
plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Loyalty score', fontsize=12)
plt.title("Feature 1 distribution")
plt.show()

# feature 2
plt.figure(figsize=(8,4))
sns.violinplot(x="feature_2", y='target', data=train_df)
plt.xticks(rotation='vertical')
plt.xlabel('Feature 2', fontsize=12)
plt.ylabel('Loyalty score', fontsize=12)
plt.title("Feature 2 distribution")
plt.show()

# feature 3
plt.figure(figsize=(8,4))
sns.violinplot(x="feature_3", y='target', data=train_df)
plt.xticks(rotation='vertical')
plt.xlabel('Feature 3', fontsize=12)
plt.ylabel('Loyalty score', fontsize=12)
plt.title("Feature 3 distribution")
plt.show()

#Add elapsed time in train and test dataset
train_df['first_active_month'] = pd.to_datetime(train_df['first_active_month'])
test_df['first_active_month'] = pd.to_datetime(test_df['first_active_month'])
train_df['elapsed_time'] = (datetime.date(2018, 2, 1) - train_df['first_active_month'].dt.date).dt.days
test_df['elapsed_time'] = (datetime.date(2018, 2, 1) - test_df['first_active_month'].dt.date).dt.days


#Load historical and new_merchant data
hist_df = pd.read_csv("historical_transactions.csv")
new_merchant_transactions= pd.read_csv("new_merchant_transactions.csv")
merchants= pd.read_csv("merchants.csv")

#unique values for each feature of historical and new_merchants
hist_df.authorized_flag.unique()
hist_df.category_1.unique()
hist_df.installments.unique()
hist_df.category_3.unique()
hist_df.category_2.unique()
hist_df.state_id.unique()
hist_df.subsector_id.unique()

new_merchant_transactions.authorized_flag.unique()
new_merchant_transactions.category_1.unique()
new_merchant_transactions.installments.unique()
new_merchant_transactions.category_3.unique()
new_merchant_transactions.category_2.unique()
new_merchant_transactions.state_id.unique()
new_merchant_transactions.subsector_id.unique()


#Handle missing values 
def missing(dataframe):
    for i in dataframe.columns:
        if dataframe[i].dtype == "object":
            dataframe[i] = dataframe[i].fillna("other")
        elif (dataframe[i].dtype == "int64" or dataframe[i].dtype == "float64" or dataframe[i].dtype == "float32"):
            dataframe[i] = dataframe[i].fillna(dataframe[i].mean())
        else:
            pass
    return dataframe
# Do impute missing values
for df in [train_df, test_df, merchants, hist_df, new_merchant_transactions]:
    missing(df)
    
#Create different features from first_Active_month
def datetime_extract(df, dt_col='first_active_month'):
    df['day'] = df[dt_col].dt.day 
    df['dayofweek'] = df[dt_col].dt.dayofweek
    df['dayofyear'] = df[dt_col].dt.dayofyear
    df['days_in_month'] = df[dt_col].dt.days_in_month
    df['daysinmonth'] = df[dt_col].dt.daysinmonth 
    df['month'] = df[dt_col].dt.month
    df['week'] = df[dt_col].dt.week 
    df['weekday'] = df[dt_col].dt.weekday
    df['weekofyear'] = df[dt_col].dt.weekofyear
    return df

# Do extract datetime values
train_df = datetime_extract(train_df, dt_col='first_active_month')
test_df = datetime_extract(test_df, dt_col='first_active_month')

#Handle categorical values of hist_df and new_merchant_transactions
def binarize_dataset(dataset):
    for column in ['authorized_flag', 'category_1']:
        dataset[column] = dataset[column].map({'Y':1, 'N':0})
    return dataset

hist_df = binarize_dataset(hist_df )
new_merchant_transactions = binarize_dataset(new_merchant_transactions)

#One hot encodding
hist_df.columns
hist_df= pd.get_dummies(hist_df, columns=['category_2', 'category_3'])
new_merchant_transactions = pd.get_dummies(new_merchant_transactions, columns=['category_2', 'category_3'])

#purchase month
type(hist_df['purchase_date'])
hist_df['purchase_month'] = pd.to_datetime(hist_df['purchase_date']).dt.month
new_merchant_transactions['purchase_month'] = pd.to_datetime(new_merchant_transactions['purchase_date']).dt.month

#For historical and new_merchant dataset
aggregations = {
    'category_1': ['sum', 'mean'],
    'category_2_1.0': ['mean'],
    'category_2_2.0': ['mean'],
    'category_2_3.0': ['mean'],
    'category_2_4.0': ['mean'],
    'category_2_5.0': ['mean'],
    'category_3_A': ['mean'],
    'category_3_B': ['mean'],
    'category_3_C': ['mean'],
    'merchant_id': ['nunique'],
    'merchant_category_id': ['nunique'],
    'state_id': ['nunique'],
    'city_id': ['nunique'],
    'subsector_id': ['nunique'],
    'purchase_amount': ['sum', 'mean', 'max', 'min', 'std'],
    'installments': ['sum', 'mean', 'max', 'min', 'std'],
    'purchase_month': ['mean', 'max', 'min', 'std'],
    'purchase_date': ['min', 'max'],
    'month_lag': ['mean', 'max', 'min', 'std']
}


#Group by card_id with the aggregation function
hist_df.loc[:, 'purchase_date'] = pd.DatetimeIndex(hist_df['purchase_date']).\
                                      astype(np.int64) * 1e-9
agg_history = hist_df.groupby(['card_id']).agg(aggregations)
agg_history.columns = ['_'.join(col).strip() for col in agg_history.columns.values] 
agg_history.reset_index(inplace=True)
agg_history.columns = ['hist_' + c if c != 'card_id' else c for c in agg_history.columns]
#36 features are extrected from hist_df

#Merge hist_df to train_df and test_df 
train_df = pd.merge(train_df, agg_history, on="card_id", how="left")
test_df = pd.merge(test_df, agg_history, on="card_id", how="left")

#Features of new_merchant_transactions at this point is 23
#Group by card_id with the aggregation function
new_merchant_transactions.loc[:, 'purchase_date'] = pd.DatetimeIndex(new_merchant_transactions['purchase_date']).\
                                      astype(np.int64) * 1e-9
agg_new_merchant= new_merchant_transactions.groupby(['card_id']).agg(aggregations)
agg_new_merchant.columns = ['_'.join(col).strip() for col in agg_new_merchant.columns.values] 
agg_new_merchant.reset_index(inplace=True)
agg_new_merchant.columns = ['new_' + c if c != 'card_id' else c for c in agg_new_merchant.columns]
#36 features are extrected from agg_new_merchant

#Merge agg_new_merchant to train_df and test_df  
train_df = pd.merge(train_df, agg_new_merchant, on="card_id", how="left")
test_df = pd.merge(test_df, agg_new_merchant, on="card_id", how="left")

#After merging the extracted 36 features to train_df, features of train_df are 87

#Unimportant features list after evaluating feature importance plot
unimportant_features = [
     'hist_category_3_A_mean',
     'new_merchant_category_id_nunique',
     'week',
     'hist_category_2_3.0_mean',
     'hist_category_2_5.0_mean',
     'hist_category_1_sum',
     'hist_category_2_4.0_mean',
     'hist_category_3_A_mean',
     'hist_category_2_2.0_mean',
     'hist_installments_min',
     'hist_installments_max',
     'hist_purchase_month_min',
     'hist_purchase_month_max',
     'hist_month_lag_min',
     'hist_month_lag_max',
     'hist_state_id_nunique',
     'seature_2',
     'new_category_1_mean',
     'new_category_2_1.0_mean',
     'new_category_2_2.0_mean',
     'new_category_2_3.0_mean',
     'new_category_2_4.0_mean',
     'new_category_2_5.0_mean',
     'new_category_3_C_mean',
     'new_category_3_A_mean',
     'new_category_3_B_mean',
     'new_city_id_nunique',
     'new_purchase_month_min',
     'new_purchase_month_max',
     'new_installments_max',
     'new_installments_min',
     'new_state_id_nunique',
     'new_month_lag_std',
     'new_installments_std',
     'new_installments_sum',
     'new_installments_mean',
     'new_subsector_id_nunique',
     'day',
     'daysinmonth',
     'new_category_1_sum',
     'weekday',
     'new_month_lag_max',
     'month',
     'days_in_month',
     'feature_3',
     'weekofyear',
     'new_month_lag_min',
     'dayofweek'
]

features = [c for c in train_df.columns if c not in ['card_id', 'first_active_month','target', 'outliers']]
features = [f for f in features if f not in unimportant_features]
categorical_feats = ['feature_1','feature_2']

# Do impute missing values
for dataframe in [train_df, test_df, merchants, hist_df, new_merchant_transactions]:
    missing(dataframe)


target = train_df['target']
del train_df['target']


#==============================================================================
#                                Model Building
#==============================================================================

#------------------------------------------------------------------------------
# Approach-1 (Regression)
# 5-Fold cross validation || rmse for regression
#------------------------------------------------------------------------------

#Linear Regression

folds = KFold(n_splits=5, shuffle=True, random_state=15)
validation_prediction = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))

from sklearn.linear_model import LinearRegression

start= time.time()
for fold_n, (train_idx, validation_idx) in enumerate(folds.split(train_df.values, target.values)):
    print("fold n{}".format(fold_n))

    train_df_x, train_df_y = train_df.iloc[train_idx][features], target.iloc[train_idx]
    valid_df_x, valid_df_y = train_df.iloc[validation_idx][features], target.iloc[validation_idx]
    
    train_df_x.dropna()
    train_df_y.dropna()
    
    regressor = LinearRegression()
    regressor.fit(train_df_x, train_df_y)
    
    validation_prediction[validation_idx] = regressor.predict(valid_df_x)
    
    predictions += regressor.predict(test_df[features]) / folds.n_splits

print("CV score: {:<8.5f}".format(mean_squared_error(validation_prediction, target)**0.5))
stop=time.time()
print("Time taken by the algorithm:", round(stop-start,2) , "seconds")


#Ridge regression
from sklearn.linear_model import Ridge

folds = KFold(n_splits=5, shuffle=True, random_state=15)
validation_prediction = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))

start = time.time()
for fold_n, (train_idx, validation_idx) in enumerate(folds.split(train_df.values,  target.values)):
    print("fold n{}".format(fold_n))

    train_df_x, train_df_y = train_df.iloc[train_idx][features], target.iloc[train_idx]
    valid_df_x, valid_df_y = train_df.iloc[validation_idx][features], target.iloc[validation_idx]
    
    
    regressor = Ridge(alpha= 100)
    regressor.fit(train_df_x, train_df_y)

    validation_prediction[validation_idx] = regressor.predict(valid_df_x)
  
    predictions += regressor.predict(test_df[features]) / folds.n_splits

print("CV score: {:<8.5f}".format(mean_squared_error(validation_prediction, target)**0.5))
stop = time.time()
print("Time taken by the algorithm:", round(stop-start, 2), "seconds")



#Decision Tree Algorithm
folds = KFold(n_splits=5, shuffle=True, random_state=15)
validation_prediction = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))

from sklearn.tree import DecisionTreeRegressor

start=time.time()
for fold_n, (train_idx, validation_idx) in enumerate(folds.split(train_df.values, target.values)):
    print("fold n{}".format(fold_n))

    train_df_x, train_df_y = train_df.iloc[train_idx][features], target.iloc[train_idx]
    valid_df_x, valid_df_y = train_df.iloc[validation_idx][features], target.iloc[validation_idx]
    
    
    regressor = DecisionTreeRegressor(random_state = 0)
    regressor.fit(train_df_x, train_df_y)

    validation_prediction[validation_idx] = regressor.predict(valid_df_x)
  
    predictions += regressor.predict(test_df[features]) / folds.n_splits

print("CV score: {:<8.5f}".format(mean_squared_error(validation_prediction, target)**0.5))
stop=time.time()
print("Time taken by the algorithm:", round(stop-start,2), "seconds")



#Random Forest
folds = KFold(n_splits=5, shuffle=True, random_state=15)
validation_prediction = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))

from sklearn.ensemble import RandomForestRegressor

start=time.time()
for fold_n, (train_idx, validation_idx) in enumerate(folds.split(train_df.values, target.values)):
    print("fold n{}".format(fold_n))

    train_df_x, train_df_y = train_df.iloc[train_idx][features], target.iloc[train_idx]
    valid_df_x, valid_df_y = train_df.iloc[validation_idx][features], target.iloc[validation_idx]
    
    
    regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
    regressor.fit(train_df_x, train_df_y)

    validation_prediction[validation_idx] = regressor.predict(valid_df_x)
    
    predictions += regressor.predict(test_df[features]) / folds.n_splits

print("CV score: {:<8.5f}".format(mean_squared_error(validation_prediction, target)**0.5))
stop= time.time()
print("Time taken by the algorithm:", round(stop-start,2), "seconds")


#SVM

from sklearn.svm import SVR

folds = KFold(n_splits=5, shuffle=True, random_state=15)
validation_prediction = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))


from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

start = time.time()
for fold_n, (train_idx, validation_idx) in enumerate(folds.split(train_df.values, target.values)):
    print("fold n{}".format(fold_n))

    train_df_x, train_df_y = train_df.iloc[train_idx][features], target.iloc[train_idx]
    valid_df_x, valid_df_y = train_df.iloc[validation_idx][features], target.iloc[validation_idx]
    
    
    regressor = SVR(kernel = 'rbf')
    regressor.fit(train_df_x, train_df_y)
    
    validation_prediction[validation_idx] = regressor.predict(valid_df_x)

    predictions += regressor.predict(test_df[features]) / folds.n_splits

print("CV score: {:<8.5f}".format(mean_squared_error(validation_prediction, target)**0.5))
stop= time.time()
print("Time taken by the algorithm:", round(stop-start,2) , "seconds")


#XGBOOST

temp_head= train_df.head()
train_df.shape
test_df.shape


import xgboost as xgb

xgb_params = {'eta': 0.005,
              'max_depth': 10,
              'subsample': 0.8,
              'colsample_bytree': 0.8,
              'objective': 'reg:linear',
              'eval_metric': 'rmse',
              'silent': True}

folds = KFold(n_splits=5, shuffle=True, random_state=15)
validation_prediction = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))

start= time.time()
for fold_n, (train_idx, validation_idx) in enumerate(folds.split(train_df, train_df['outliers'].values)):
    print("fold n{}".format(fold_n))
    
    train_df_x, train_df_y = train_df.iloc[train_idx][features], target.iloc[train_idx]
    valid_df_x, valid_df_y = train_df.iloc[validation_idx][features], target.iloc[validation_idx]
    
    trn_data = xgb.DMatrix(data=train_df_x,
                           label=train_df_y
                          )
    
    val_data = xgb.DMatrix(data=valid_df_x,
                           label=valid_df_y
                          )

    watchlist= [(trn_data, 'train'), (val_data, 'valid')]
    print("xgb " + str(fold_n) + "-" * 50)
    num_round = 10000
                          
    xgb_model = xgb.train(xgb_params,
                          trn_data,
                          num_round,
                          watchlist,
                          early_stopping_rounds = 50,
                          verbose_eval=1000
                          )
    
    validation_prediction[validation_idx] = xgb_model.predict(xgb.DMatrix(valid_df_x), ntree_limit=xgb_model.best_ntree_limit+50)
    
    predictions += xgb_model.predict(xgb.DMatrix(test_df[features]), ntree_limit=xgb_model.best_ntree_limit+50) / folds.n_splits

print("CV score: {:<8.5f}".format(mean_squared_error(validation_prediction, target)**0.5))
stop= time.time()
print("Time taken by the algorithm:", round(stop-start,2), "seconds")



#LGBM algorithm

param = {'num_leaves': 111,
         'min_data_in_leaf': 149, 
         'objective':'regression',
         'max_depth': 9,
         'learning_rate': 0.005,
         "boosting": "gbdt",
         "feature_fraction": 0.7522,
         "bagging_freq": 1,
         "bagging_fraction": 0.7083 ,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.2634,
         "random_state": 133,
         "verbosity": -1}

folds = KFold(n_splits=5, shuffle=True, random_state=15)
validation_prediction = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))
feature_importance_df = pd.DataFrame()

start= time.time()
for fold_n, (train_idx, validation_idx) in enumerate(folds.split(train_df.values, target.values)):
    print("fold n{}".format(fold_n))

    train_df_x, train_df_y = train_df.iloc[train_idx][features], target.iloc[train_idx]
    valid_df_x, valid_df_y = train_df.iloc[validation_idx][features], target.iloc[validation_idx]
    
    trn_data = lgb.Dataset(train_df_x,
                           label=train_df_y,
                           categorical_feature=categorical_feats
                          )
    
    val_data = lgb.Dataset(valid_df_x,
                           label=valid_df_y,
                           categorical_feature=categorical_feats
                          )

    num_round = 10000
    regressor = lgb.train(param,
                          trn_data,
                          num_round,
                          valid_sets = [trn_data, val_data],
                          verbose_eval=100,
                          early_stopping_rounds = 200)
    
    validation_prediction[validation_idx] = regressor.predict(valid_df_x, num_iteration=regressor.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = regressor.feature_importance()
    fold_importance_df["fold_n"] = fold_n + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += regressor.predict(test_df[features], num_iteration=regressor.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(mean_squared_error(validation_prediction, target)**0.5))
stop= time.time()
print("Time taken by the algorithm:", round(stop-start,2), "seconds")

#Importance of each feature (Plotting)
cols = (feature_importance_df[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:1000].index)

best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

plt.figure(figsize=(14,25))
sns.barplot(x="importance",
            y="feature",
            data=best_features.sort_values(by="importance",
                                           ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances.png')

#Evaluate first few prediction (Plotting)
temp_target=[]
for i in target:
    temp_target.append(i)    
temp_validation_prediction=[]
for i in validation_prediction:
    temp_validation_prediction.append(i)
    
df=pd.DataFrame({'x': range(20), 
                 'y1': temp_target[:20], 
                 'y2': temp_validation_prediction[:20]
                 })
 
# multiple line plot
plt.plot( 'x', 'y1', data=df, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
plt.plot( 'x', 'y2', data=df, marker='', color='olive', linewidth=2)
plt.legend()


#Deep Learning (Artificial NN tried but it failed totally)

#------------------------------------------------------------------------------
# Approach-2 (Classification)
# 5-Fold cross validation || Confusion Matrix (F1-score)
#------------------------------------------------------------------------------
train_df.loc[train_df['target'] <= 0.0, 'target'] = 0
train_df.loc[train_df['target'] > 0.0, 'target'] = 1

train_df.target.describe()

#Decision Tree Algorithm
folds = KFold(n_splits=5, shuffle=True, random_state=15)
validation_prediction = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))

from sklearn.tree import DecisionTreeClassifier

start=time.time()
for fold_n, (train_idx, validation_idx) in enumerate(folds.split(train_df.values, target.values)):
    print("fold n{}".format(fold_n))

    train_df_x, train_df_y = train_df.iloc[train_idx][features], target.iloc[train_idx]
    valid_df_x, valid_df_y = train_df.iloc[validation_idx][features], target.iloc[validation_idx]
    
    
    regressor = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    regressor.fit(train_df_x, train_df_y)

    validation_prediction[validation_idx] = regressor.predict(valid_df_x)
  
    predictions += regressor.predict(test_df[features]) / folds.n_splits

print("CV score: {:<8.5f}".format(mean_squared_error(validation_prediction, target)**0.5))
stop=time.time()
print("Time taken by the algorithm:", round(stop-start,2), "seconds")

from sklearn.metrics import confusion_matrix
cm0 = confusion_matrix(target, validation_prediction)

tn, fp, fn, tp= cm0.ravel()
prec = tp/(tp+fp)
rec = tp/(tp+fn)
F1 = (2*prec*rec)/(prec+rec)


#Random forest Algorithm
folds = KFold(n_splits=5, shuffle=True, random_state=15)
validation_prediction = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))

from sklearn.ensemble import RandomForestClassifier

start=time.time()
for fold_n, (train_idx, validation_idx) in enumerate(folds.split(train_df.values, target.values)):
    print("fold n{}".format(fold_n))

    train_df_x, train_df_y = train_df.iloc[train_idx][features], target.iloc[train_idx]
    valid_df_x, valid_df_y = train_df.iloc[validation_idx][features], target.iloc[validation_idx]
    
    
    regressor = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    regressor.fit(train_df_x, train_df_y)

    validation_prediction[validation_idx] = regressor.predict(valid_df_x)
  
    predictions += regressor.predict(test_df[features]) / folds.n_splits

print("CV score: {:<8.5f}".format(mean_squared_error(validation_prediction, target)**0.5))
stop=time.time()
print("Time taken by the algorithm:", round(stop-start,2), "seconds")

from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(target, validation_prediction)

tn, fp, fn, tp= cm1.ravel()
prec = tp/(tp+fp)
rec = tp/(tp+fn)
F1 = (2*prec*rec)/(prec+rec)


#Naive Bayes
folds = KFold(n_splits=5, shuffle=True, random_state=15)
validation_prediction = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))

from sklearn.naive_bayes import GaussianNB

start=time.time()
for fold_n, (train_idx, validation_idx) in enumerate(folds.split(train_df.values, target.values)):
    print("fold n{}".format(fold_n))

    train_df_x, train_df_y = train_df.iloc[train_idx][features], target.iloc[train_idx]
    valid_df_x, valid_df_y = train_df.iloc[validation_idx][features], target.iloc[validation_idx]
    
    
    regressor = GaussianNB()
    regressor.fit(train_df_x, train_df_y)

    validation_prediction[validation_idx] = regressor.predict(valid_df_x)
    
    predictions += regressor.predict(test_df[features]) / folds.n_splits

print("CV score: {:<8.5f}".format(mean_squared_error(validation_prediction, target)**0.5))
stop= time.time()
print("Time taken by the algorithm:", round(stop-start,2), "seconds")

from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(target, validation_prediction)

tn, fp, fn, tp= cm2.ravel()
prec = tp/(tp+fp)
rec = tp/(tp+fn)
F1 = (2*prec*rec)/(prec+rec)


#Knn
folds = KFold(n_splits=5, shuffle=True, random_state=15)
validation_prediction = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))

from sklearn.neighbors import KNeighborsClassifier

start=time.time()
for fold_n, (train_idx, validation_idx) in enumerate(folds.split(train_df.values, target.values)):
    print("fold n{}".format(fold_n))

    train_df_x, train_df_y = train_df.iloc[train_idx][features], target.iloc[train_idx]
    valid_df_x, valid_df_y = train_df.iloc[validation_idx][features], target.iloc[validation_idx]
    
    
    regressor = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    regressor.fit(train_df_x, train_df_y)

    validation_prediction[validation_idx] = regressor.predict(valid_df_x)
    
    predictions += regressor.predict(test_df[features]) / folds.n_splits

print("CV score: {:<8.5f}".format(mean_squared_error(validation_prediction, target)**0.5))
stop= time.time()
print("Time taken by the algorithm:", round(stop-start,2), "seconds")

cm3 = confusion_matrix(target, validation_prediction)

tn, fp, fn, tp= cm3.ravel()
prec = tp/(tp+fp)
rec = tp/(tp+fn)
F1 = (2*prec*rec)/(prec+rec)


#Logistic Regression
folds = KFold(n_splits=5, shuffle=True, random_state=15)
validation_prediction = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))

from sklearn.linear_model import LogisticRegression

start=time.time()
for fold_n, (train_idx, validation_idx) in enumerate(folds.split(train_df.values, target.values)):
    print("fold n{}".format(fold_n))

    train_df_x, train_df_y = train_df.iloc[train_idx][features], target.iloc[train_idx]
    valid_df_x, valid_df_y = train_df.iloc[validation_idx][features], target.iloc[validation_idx]
    
    
    regressor = LogisticRegression(random_state = 0)
    regressor.fit(train_df_x, train_df_y)

    validation_prediction[validation_idx] = regressor.predict(valid_df_x)
    
    predictions += regressor.predict(test_df[features]) / folds.n_splits

print("CV score: {:<8.5f}".format(mean_squared_error(validation_prediction, target)**0.5))
stop= time.time()
print("Time taken by the algorithm:", round(stop-start,2), "seconds")

cm4 = confusion_matrix(target, validation_prediction)

tn, fp, fn, tp= cm4.ravel()
prec = tp/(tp+fp)
rec = tp/(tp+fn)
F1 = (2*prec*rec)/(prec+rec)


#Kernel SVM
folds = KFold(n_splits=5, shuffle=True, random_state=15)
validation_prediction = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))

from sklearn.svm import SVC

start=time.time()
for fold_n, (train_idx, validation_idx) in enumerate(folds.split(train_df.values, target.values)):
    print("fold n{}".format(fold_n))

    train_df_x, train_df_y = train_df.iloc[train_idx][features], target.iloc[train_idx]
    valid_df_x, valid_df_y = train_df.iloc[validation_idx][features], target.iloc[validation_idx]
    
    
    regressor = SVC(kernel = 'rbf', random_state = 0)
    regressor.fit(train_df_x, train_df_y)

    validation_prediction[validation_idx] = regressor.predict(valid_df_x)
    
    predictions += regressor.predict(test_df[features]) / folds.n_splits

print("CV score: {:<8.5f}".format(mean_squared_error(validation_prediction, target)**0.5))
stop= time.time()
print("Time taken by the algorithm:", round(stop-start,2), "seconds")

cm5 = confusion_matrix(target, validation_prediction)

tn, fp, fn, tp= cm5.ravel()
prec = tp/(tp+fp)
rec = tp/(tp+fn)
F1 = (2*prec*rec)/(prec+rec)


#XGBOOST
temp_head= train_df.head()
train_df.shape
test_df.shape


import xgboost as xgb

xgb_params = {'eta': 0.005,
              'max_depth': 10,
              'subsample': 0.8,
              'colsample_bytree': 0.8,
              'objective': 'binary:logistic',
              'eval_metric': 'error',
              'silent': True}

folds = KFold(n_splits=5, shuffle=True, random_state=15)
validation_prediction = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))

start= time.time()
for fold_n, (train_idx, validation_idx) in enumerate(folds.split(train_df, train_df['outliers'].values)):
    print("fold n{}".format(fold_n))
    
    train_df_x, train_df_y = train_df.iloc[train_idx][features], target.iloc[train_idx]
    valid_df_x, valid_df_y = train_df.iloc[validation_idx][features], target.iloc[validation_idx]
    
    trn_data = xgb.DMatrix(data=train_df_x,
                           label=train_df_y
                          )
    
    val_data = xgb.DMatrix(data=valid_df_x,
                           label=valid_df_y
                          )

    watchlist= [(trn_data, 'train'), (val_data, 'valid')]
    print("xgb " + str(fold_n) + "-" * 50)
    num_round = 10000
                          
    xgb_model = xgb.train(xgb_params,
                          trn_data,
                          num_round,
                          watchlist,
                          early_stopping_rounds = 50,
                          verbose_eval=1000
                          )
    
    validation_prediction[validation_idx] = xgb_model.predict(xgb.DMatrix(valid_df_x), ntree_limit=xgb_model.best_ntree_limit+50)
    
    predictions += xgb_model.predict(xgb.DMatrix(test_df[features]), ntree_limit=xgb_model.best_ntree_limit+50) / folds.n_splits

for i in range(len(validation_prediction)):
    if validation_prediction[i]>=.5:       # setting threshold to .5
       validation_prediction[i]=1
    else:  
       validation_prediction[i]=0

print("CV score: {:<8.5f}".format(mean_squared_error(validation_prediction, target)**0.5))
stop= time.time()
print("Time taken by the algorithm:", round(stop-start,2), "seconds")

cm7 = confusion_matrix(target, validation_prediction)


tn, fp, fn, tp= cm7.ravel()
prec = tp/(tp+fp)
rec = tp/(tp+fn)
F1 = (2*prec*rec)/(prec+rec)



#LGBM algorithm
param = {'num_leaves': 111,
         'min_data_in_leaf': 149, 
         'objective':'binary',
         'max_depth': 9,
         'learning_rate': 0.005,
         "boosting": "gbdt",
         "feature_fraction": 0.7522,
         "bagging_freq": 1,
         "bagging_fraction": 0.7083 ,
         "bagging_seed": 11,
         "metric": 'binary_logloss',
         "lambda_l1": 0.2634,
         "random_state": 133,
         "verbosity": -1}

folds = KFold(n_splits=5, shuffle=True, random_state=15)
validation_prediction = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))
feature_importance_df = pd.DataFrame()

start= time.time()
for fold_n, (train_idx, validation_idx) in enumerate(folds.split(train_df.values, target.values)):
    print("fold n{}".format(fold_n))

    train_df_x, train_df_y = train_df.iloc[train_idx][features], target.iloc[train_idx]
    valid_df_x, valid_df_y = train_df.iloc[validation_idx][features], target.iloc[validation_idx]
    
    trn_data = lgb.Dataset(train_df_x,
                           label=train_df_y,
                           categorical_feature=categorical_feats
                          )
    
    val_data = lgb.Dataset(valid_df_x,
                           label=valid_df_y,
                           categorical_feature=categorical_feats
                          )

    num_round = 10000
    regressor = lgb.train(param,
                          trn_data,
                          num_round,
                          valid_sets = [trn_data, val_data],
                          verbose_eval=100,
                          early_stopping_rounds = 200)
    
    validation_prediction[validation_idx] = regressor.predict(valid_df_x, num_iteration=regressor.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = regressor.feature_importance()
    fold_importance_df["fold_n"] = fold_n + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += regressor.predict(test_df[features], num_iteration=regressor.best_iteration) / folds.n_splits

for i in range(len(validation_prediction)):
    if validation_prediction[i]>=.5:       # setting threshold to .5
       validation_prediction[i]=1
    else:  
       validation_prediction[i]=0
       
print("CV score: {:<8.5f}".format(mean_squared_error(validation_prediction, target)**0.5))
stop= time.time()
print("Time taken by the algorithm:", round(stop-start,2), "seconds")

cm6 = confusion_matrix(target, validation_prediction)

tn, fp, fn, tp= cm6.ravel()
prec = tp/(tp+fp)
rec = tp/(tp+fn)
F1 = (2*prec*rec)/(prec+rec)


#Plotting
temp_target=[]
for i in target:
    temp_target.append(i)    
temp_validation_prediction=[]
for i in validation_prediction:
    temp_validation_prediction.append(i)
    
df=pd.DataFrame({'x': range(20), 
                 'y1': temp_target[:20], 
                 'y2': temp_validation_prediction[:20]
                 })
 
# multiple line plot
plt.plot( 'x', 'y1', data=df, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
plt.plot( 'x', 'y2', data=df, marker='', color='olive', linewidth=2)
plt.legend()


