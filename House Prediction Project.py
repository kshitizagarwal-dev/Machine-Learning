#!/usr/bin/env python
# coding: utf-8

# ### Importing the modules

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston


# ### Loading Dataset

# In[2]:


boston_dataset = load_boston()


# ### EDA 

# In[3]:


type(boston_dataset)


# In[4]:


boston_dataset


# In[5]:


boston_dataset_X = boston_dataset['data']
boston_dataset_Y = boston_dataset['target']
boston_dataset_X_names = boston_dataset['feature_names']
boston_dataset_X_desc = boston_dataset['DESCR']


# In[6]:


print(boston_dataset_X_desc)


# #### Creating the DataFrame

# In[7]:


def get_df_info(df, include_unique_values=False):
    col_name_list = list(df.columns)
    col_type_list = [type(df[col][0]) for col in col_name_list]
    col_null_count_list = [df[col].isnull().sum() for col in col_name_list]
    col_unique_count_list = [df[col].nunique() for col in col_name_list]
    col_memory_usage_list = [df[col].memory_usage(deep=True) for col in col_name_list]
    df_total_memory_usage = sum(col_memory_usage_list) / 1048576
    if include_unique_values:
        col_unique_list = [df[col].unique() for col in col_name_list]
        df_info = pd.DataFrame({'column_name': col_name_list, 'type': col_type_list, 
                                'null_count': col_null_count_list, 'nunique': col_unique_count_list, 
                                'unique_values': col_unique_list})
    else:
        df_info = pd.DataFrame({'column_name': col_name_list, 'type': col_type_list, 
                                'null_count': col_null_count_list, 'nunique': col_unique_count_list})
    return df_info, df_total_memory_usage


# In[8]:


df_boston_data = pd.DataFrame(data =boston_dataset_X, columns = boston_dataset_X_names )
df_boston_data['MEDV'] = boston_dataset_Y
df_boston_data.head()


# In[9]:


df_boston_data_info, df_boston_data_mem_usage = get_df_info(df_boston_data, True)
df_boston_data_info


# In[10]:


df_boston_data_mem_usage


# In[11]:


df_boston_data.describe()


# ### PreProcessing the DataFrame

# In[12]:


df_boston_data.CHAS = df_boston_data.CHAS.astype(np.int8)
df_boston_data.RAD = df_boston_data.RAD.astype(np.int8)
df_boston_data.PTRATIO =df_boston_data.PTRATIO.astype(np.float32)


# In[13]:


df_boston_data_info, df_boston_data_mem_usage = get_df_info(df_boston_data, True)
df_boston_data_info


# #### Data Visualization

# In[14]:


import seaborn as sb
sb.set(rc = {'figure.figsize' : (14, 9)} )
sb.heatmap(df_boston_data.corr(), cmap = 'coolwarm', vmax = -1, vmin =  1, annot = True)


# ### Create Models and Evaluate them 

# In[15]:


df_boston_data.hist()


# In[16]:


from sklearn.model_selection import cross_validate, train_test_split
from sklearn.linear_model import LinearRegression as LR 
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.ensemble import RandomForestRegressor as RFR

from sklearn.metrics import r2_score, mean_absolute_error


# In[17]:


def show_model_eval_table(model_attrib):
    df_model_eval = pd.DataFrame({'model': model_attrib['model_names'],
                                  'feature_count': model_attrib['model_feature_counts'], 
                                  'feature_names': model_attrib['model_feature_names'], 
                                  'r2': model_attrib['model_r2_scores'], 
                                  'mae': model_attrib['model_mae_scores']})
    return df_model_eval.round(2)


# In[18]:


model_attrib = {
    'model_names': [],
    'model_feature_counts': [],
    'model_feature_names': [],
    'model_r2_scores': [],
    'model_mae_scores': []
}


# ### Creating model 1 with all features.

# In[19]:


X_train, X_test, Y_train, Y_test= train_test_split(df_boston_data.loc[:, :'LSTAT' ], df_boston_data.MEDV, random_state = 0)
print(X_train.shape, X_test.shape)
print(Y_train.shape, Y_test.shape)


# In[20]:


### Linear Regression model
lr_model_1 = LR()
lr_model_1.fit(X_train, Y_train)
y_hat_lr_model_1 = lr_model_1.predict(X_test)
model_attrib['model_names'].append('lr_model_1')
model_attrib['model_feature_counts'].append(X_train.shape[1])
model_attrib['model_feature_names'].append(list(X_train.columns))
model_attrib['model_r2_scores'].append(r2_score(Y_test, y_hat_lr_model_1))
model_attrib['model_mae_scores'].append(mean_absolute_error(Y_test, y_hat_lr_model_1))

#RandomForestRegressor
rf_model_1 = RFR(random_state = 0)
rf_model_1.fit(X_train, Y_train)
y_hat_rf_model_1 = rf_model_1.predict(X_test)
model_attrib['model_names'].append('rf_model_1')
model_attrib['model_feature_counts'].append(X_train.shape[1])
model_attrib['model_feature_names'].append(list(X_train.columns))
model_attrib['model_r2_scores'].append(r2_score(Y_test, y_hat_rf_model_1))
model_attrib['model_mae_scores'].append(mean_absolute_error(Y_test, y_hat_rf_model_1))


#DecisionTreeRegressor
dt_model_1 = DTR(random_state = 0)
dt_model_1.fit(X_train, Y_train)
y_hat_dt_model_1 = dt_model_1.predict(X_test)
model_attrib['model_names'].append('dt_model_1')
model_attrib['model_feature_counts'].append(X_train.shape[1])
model_attrib['model_feature_names'].append(list(X_train.columns))
model_attrib['model_r2_scores'].append(r2_score(Y_test, y_hat_dt_model_1))
model_attrib['model_mae_scores'].append(mean_absolute_error(Y_test, y_hat_dt_model_1))


# In[21]:


show_model_eval_table(model_attrib)


# ### Creating X2  set using the correlation technique with 10 features

# In[22]:


X2_train = X_train.drop(["CHAS", "RAD", 'DIS'], axis = 1)
X2_test = X_test.drop(["CHAS", "RAD", 'DIS'], axis = 1)
print(X2_train.shape, X2_test.shape)
print(Y_train.shape, Y_test.shape)


# ### Creating model 2 with X2 features.

# In[23]:


### Linear Regression model
lr_model_2 = LR()
lr_model_2.fit(X2_train, Y_train)
y_hat_lr_model_2 = lr_model_2.predict(X2_test)
model_attrib['model_names'].append('lr_model_2')
model_attrib['model_feature_counts'].append(X2_train.shape[1])
model_attrib['model_feature_names'].append(list(X2_train.columns))
model_attrib['model_r2_scores'].append(r2_score(Y_test, y_hat_lr_model_2))
model_attrib['model_mae_scores'].append(mean_absolute_error(Y_test, y_hat_lr_model_2))

#RandomForestRegressor
rf_model_2 = RFR(random_state = 0)
rf_model_2.fit(X2_train, Y_train)
y_hat_rf_model_2 = rf_model_2.predict(X2_test)
model_attrib['model_names'].append('rf_model_2')
model_attrib['model_feature_counts'].append(X2_train.shape[1])
model_attrib['model_feature_names'].append(list(X2_train.columns))
model_attrib['model_r2_scores'].append(r2_score(Y_test, y_hat_rf_model_2))
model_attrib['model_mae_scores'].append(mean_absolute_error(Y_test, y_hat_rf_model_2))


#DecisionTreeRegressor
dt_model_2 = DTR(random_state = 0)
dt_model_2.fit(X2_train, Y_train)
y_hat_dt_model_2 = dt_model_2.predict(X2_test)
model_attrib['model_names'].append('dt_model_2')
model_attrib['model_feature_counts'].append(X2_train.shape[1])
model_attrib['model_feature_names'].append(list(X2_train.columns))
model_attrib['model_r2_scores'].append(r2_score(Y_test, y_hat_dt_model_2))
model_attrib['model_mae_scores'].append(mean_absolute_error(Y_test, y_hat_dt_model_2))


# In[24]:


show_model_eval_table(model_attrib)


# ### Creating the feature set XS with StandardScaler

# In[25]:


from sklearn.preprocessing import StandardScaler


# In[26]:


std_scaler = StandardScaler()
XS  = std_scaler.fit_transform(df_boston_data.iloc[:, :-1].values)


# In[27]:


XS = pd.DataFrame(XS, columns=boston_dataset_X_names)
XS.head()


# In[28]:


XS_train, XS_test = train_test_split(XS, random_state=0)
print(XS_train.shape, Y_train.shape)
print(XS_test.shape, Y_test.shape)


# In[29]:


XS.describe().round(2)


# ### Creating model3 with all XS feature set.

# In[30]:


### Linear Regression model
lr_model_3 = LR()
lr_model_3.fit(XS_train, Y_train)
y_hat_lr_model_3 = lr_model_3.predict(XS_test)
model_attrib['model_names'].append('lr_model_3')
model_attrib['model_feature_counts'].append(XS_train.shape[1])
model_attrib['model_feature_names'].append(list(X_train.columns))
model_attrib['model_r2_scores'].append(r2_score(Y_test, y_hat_lr_model_3))
model_attrib['model_mae_scores'].append(mean_absolute_error(Y_test, y_hat_lr_model_3))


# In[31]:


show_model_eval_table(model_attrib)


# In[32]:


rf_model_3 = RFR(random_state = 0)
rf_model_3.fit(XS_train, Y_train)
y_hat_rf_model_3 = rf_model_3.predict(XS_test)
model_attrib['model_names'].append('rf_model_3')
model_attrib['model_feature_counts'].append(XS_train.shape[1])
model_attrib['model_feature_names'].append(list(X_train.columns))
model_attrib['model_r2_scores'].append(r2_score(Y_test, y_hat_rf_model_3))
model_attrib['model_mae_scores'].append(mean_absolute_error(Y_test, y_hat_rf_model_3))


# In[33]:


show_model_eval_table(model_attrib)


# In[34]:


dt_model_3 = DTR(random_state = 0)
dt_model_3.fit(XS_train, Y_train)
y_hat_dt_model_3 = dt_model_3.predict(XS_test)
model_attrib['model_names'].append('dt_model_3')
model_attrib['model_feature_counts'].append(XS_train.shape[1])
model_attrib['model_feature_names'].append(list(X_train.columns))
model_attrib['model_r2_scores'].append(r2_score(Y_test, y_hat_dt_model_3))
model_attrib['model_mae_scores'].append(mean_absolute_error(Y_test, y_hat_dt_model_3))


# In[35]:


show_model_eval_table(model_attrib)


# ### Creating the feature set X2S with 10 features using StandardScaler

# In[36]:


X2S = XS.drop(["CHAS", 'RAD', 'DIS'], axis = 1)
print(X2S.shape)


# In[37]:


X2S_train = XS_train.drop(["CHAS", 'RAD', 'DIS'], axis = 1)
X2S_test = XS_test.drop(["CHAS", 'RAD', 'DIS'], axis = 1)
print(X2S_train.shape, X2S_test.shape)


# In[38]:


X2S_train.describe().round(2)


# ### Creating model 4 with  X2S feature set.

# In[39]:


### Linear Regression model
lr_model_4 = LR()
lr_model_4.fit(X2S_train, Y_train)
y_hat_lr_model_4 = lr_model_4.predict(X2S_test)
model_attrib['model_names'].append('lr_model_4')
model_attrib['model_feature_counts'].append(X2S_train.shape[1])
model_attrib['model_feature_names'].append(list(X2_train.columns))
model_attrib['model_r2_scores'].append(r2_score(Y_test, y_hat_lr_model_4))
model_attrib['model_mae_scores'].append(mean_absolute_error(Y_test, y_hat_lr_model_4))


# In[40]:


show_model_eval_table(model_attrib)


# In[41]:


rf_model_4 = RFR(random_state  = 0)
rf_model_4.fit(X2S_train, Y_train)
y_hat_rf_model_4 = rf_model_4.predict(X2S_test)
model_attrib['model_names'].append('rf_model_4')
model_attrib['model_feature_counts'].append(X2S_train.shape[1])
model_attrib['model_feature_names'].append(list(X2_train.columns))
model_attrib['model_r2_scores'].append(r2_score(Y_test, y_hat_rf_model_4))
model_attrib['model_mae_scores'].append(mean_absolute_error(Y_test, y_hat_rf_model_4))


# In[42]:


dt_model_4 = DTR(random_state = 0)
dt_model_4.fit(X2S_train, Y_train)
y_hat_dt_model_4 = dt_model_4.predict(X2S_test)
model_attrib['model_names'].append('dt_model_4')
model_attrib['model_feature_counts'].append(X2S_train.shape[1])
model_attrib['model_feature_names'].append(list(X2_train.columns))
model_attrib['model_r2_scores'].append(r2_score(Y_test, y_hat_dt_model_4))
model_attrib['model_mae_scores'].append(mean_absolute_error(Y_test, y_hat_dt_model_4))


# In[43]:


show_model_eval_table(model_attrib)


# ### HyperTuning the parameters for the X2S data set.

# In[44]:


from sklearn.model_selection import GridSearchCV


# In[45]:


hp_rf_grid = {'n_estimators' : [10, 20, 50, 100, 200, 500],
                              'max_features':['auto', 'sqrt', 'log2']}
hp_dt_grid = {'max_features': ['auto', 'sqrt', 'log2', None],
                             'splitter' :['best', 'random']}

gscv_rf = GridSearchCV(RFR(random_state = 0), param_grid =hp_rf_grid, n_jobs = 5, cv = 5, verbose = 10 )
gscv_rf.fit(X2S, df_boston_data.MEDV)


# In[46]:


gscv_rf.best_params_


# ### Creating model 5 with Hyperparameter tuning

# In[47]:


rf_model_5 = RFR(random_state  = 0, max_features = 'sqrt', n_estimators = 500)
rf_model_5.fit(X2S_train, Y_train)
y_hat_rf_model_5 = rf_model_5.predict(X2S_test)
model_attrib['model_names'].append('rf_model_5')
model_attrib['model_feature_counts'].append(X2S_train.shape[1])
model_attrib['model_feature_names'].append(list(X2S_train.columns))
model_attrib['model_r2_scores'].append(r2_score(Y_test, y_hat_rf_model_5))
model_attrib['model_mae_scores'].append(mean_absolute_error(Y_test, y_hat_rf_model_5))


# In[48]:


show_model_eval_table(model_attrib)


# In[49]:


gscv_dt = GridSearchCV(DTR(random_state = 0), param_grid =hp_dt_grid, n_jobs = 5, cv = 5, verbose = 10 )
gscv_dt.fit(X2S, df_boston_data.MEDV)


# In[50]:


gscv_dt.best_params_


# In[51]:


dt_model_5 = DTR(random_state = 0, max_features =  'sqrt', splitter = 'best')
dt_model_5.fit(X2S_train, Y_train)
y_hat_dt_model_5 = dt_model_5.predict(X2S_test)
model_attrib['model_names'].append('dt_model_5')
model_attrib['model_feature_counts'].append(X2S_train.shape[1])
model_attrib['model_feature_names'].append(list(X2_train.columns))
model_attrib['model_r2_scores'].append(r2_score(Y_test, y_hat_dt_model_5))
model_attrib['model_mae_scores'].append(mean_absolute_error(Y_test, y_hat_dt_model_5))


# In[52]:


show_model_eval_table(model_attrib)


# ### Creating X3 feature set using corrleation.

# In[53]:


X3 = df_boston_data.loc[:, ['RM', 'LSTAT', 'PTRATIO']]
print(X3.shape)


# In[54]:


X3_train = X_train.loc[:, ['RM', 'LSTAT', 'PTRATIO']]
X3_test= X_test.loc[:, ['RM', 'LSTAT', 'PTRATIO']]
print(X3_train.shape, X3_test.shape)


# ### Creating model 6 with X3 features.

# In[55]:


### Linear Regression model
lr_model_6 = LR()
lr_model_6.fit(X3_train, Y_train)
y_hat_lr_model_6 = lr_model_6.predict(X3_test)
model_attrib['model_names'].append('lr_model_6')
model_attrib['model_feature_counts'].append(X3_train.shape[1])
model_attrib['model_feature_names'].append(list(X3_train.columns))
model_attrib['model_r2_scores'].append(r2_score(Y_test, y_hat_lr_model_6))
model_attrib['model_mae_scores'].append(mean_absolute_error(Y_test, y_hat_lr_model_6))

#RandomForestRegressor
rf_model_6 = RFR(random_state = 0)
rf_model_6.fit(X3_train, Y_train)
y_hat_rf_model_6 = rf_model_6.predict(X3_test)
model_attrib['model_names'].append('rf_model_6')
model_attrib['model_feature_counts'].append(X3_train.shape[1])
model_attrib['model_feature_names'].append(list(X3_train.columns))
model_attrib['model_r2_scores'].append(r2_score(Y_test, y_hat_rf_model_6))
model_attrib['model_mae_scores'].append(mean_absolute_error(Y_test, y_hat_rf_model_6))


#DecisionTreeRegressor
dt_model_6 = DTR(random_state = 0)
dt_model_6.fit(X3_train, Y_train)
y_hat_dt_model_6 = dt_model_6.predict(X3_test)
model_attrib['model_names'].append('dt_model_6')
model_attrib['model_feature_counts'].append(X3_train.shape[1])
model_attrib['model_feature_names'].append(list(X3_train.columns))
model_attrib['model_r2_scores'].append(r2_score(Y_test, y_hat_dt_model_6))
model_attrib['model_mae_scores'].append(mean_absolute_error(Y_test, y_hat_dt_model_6))


# In[56]:


show_model_eval_table(model_attrib)


# In[57]:


X3S = X2S.loc[:,  ['RM', 'LSTAT', 'PTRATIO']]
print(X3S.shape)


# In[58]:


X3S_train = X2S_train.loc[:, ['RM', 'LSTAT', 'PTRATIO']]
X3S_test= X2S_test.loc[:, ['RM', 'LSTAT', 'PTRATIO']]
print(X3S_train.shape, X3S_test.shape)


# ### Creating model 7 using X3S feature set.

# In[59]:


### Linear Regression model
lr_model_7 = LR()
lr_model_7.fit(X3S_train, Y_train)
y_hat_lr_model_7 = lr_model_7.predict(X3S_test)
model_attrib['model_names'].append('lr_model_7')
model_attrib['model_feature_counts'].append(X3S_train.shape[1])
model_attrib['model_feature_names'].append(list(X3S_train.columns))
model_attrib['model_r2_scores'].append(r2_score(Y_test, y_hat_lr_model_7))
model_attrib['model_mae_scores'].append(mean_absolute_error(Y_test, y_hat_lr_model_7))

#RandomForestRegressor
rf_model_7 = RFR(random_state = 0)
rf_model_7.fit(X3S_train, Y_train)
y_hat_rf_model_7 = rf_model_7.predict(X3S_test)
model_attrib['model_names'].append('rf_model_7')
model_attrib['model_feature_counts'].append(X3S_train.shape[1])
model_attrib['model_feature_names'].append(list(X3S_train.columns))
model_attrib['model_r2_scores'].append(r2_score(Y_test, y_hat_rf_model_7))
model_attrib['model_mae_scores'].append(mean_absolute_error(Y_test, y_hat_rf_model_7))


#DecisionTreeRegressor
dt_model_7 = DTR(random_state = 0)
dt_model_7.fit(X3S_train, Y_train)
y_hat_dt_model_7 = dt_model_7.predict(X3S_test)
model_attrib['model_names'].append('dt_model_7')
model_attrib['model_feature_counts'].append(X3S_train.shape[1])
model_attrib['model_feature_names'].append(list(X3S_train.columns))
model_attrib['model_r2_scores'].append(r2_score(Y_test, y_hat_dt_model_7))
model_attrib['model_mae_scores'].append(mean_absolute_error(Y_test, y_hat_dt_model_7))


# In[60]:


show_model_eval_table(model_attrib)


# In[61]:


rf_model_8 = RFR(random_state  = 0, max_features = 'sqrt', n_estimators = 500)
rf_model_8.fit(XS_train, Y_train)
y_hat_rf_model_8 = rf_model_8.predict(XS_test)
model_attrib['model_names'].append('rf_model_8')
model_attrib['model_feature_counts'].append(XS_train.shape[1])
model_attrib['model_feature_names'].append(list(X_train.columns))
model_attrib['model_r2_scores'].append(r2_score(Y_test, y_hat_rf_model_8))
model_attrib['model_mae_scores'].append(mean_absolute_error(Y_test, y_hat_rf_model_8))


# In[62]:


show_model_eval_table(model_attrib)


# In[63]:


### Hence , rf_model_3 is best suitated for this dataset that is with all features and StandardScaler. 

