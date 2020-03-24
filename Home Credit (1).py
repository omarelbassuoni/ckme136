#!/usr/bin/env python
# coding: utf-8

# # Data Prep

# In[1]:


# numpy and pandas for data manipulation
import numpy as np
import pandas as pd 

# File system manangement
import os

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns

#pandas profiling for EDA
from pandas_profiling import ProfileReport
import gc

# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
#from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OrdinalEncoder 
from sklearn.experimental import enable_iterative_imputer  # noqa
# now you can import normally from sklearn.impute
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor


# In[2]:


# Read in train data
app_train = pd.read_csv('C:/Users/User/Desktop/home-credit-default-risk/application_train.csv')
# Read in test data 
app_test = pd.read_csv('C:/Users/User/Desktop/home-credit-default-risk/application_test.csv')


# # EDA

# In[3]:


#Data frame at a glance
app_train.info()


# In[4]:


#profile report
trainprofile = ProfileReport(app_train, minimal=True)
trainprofile.to_widgets()


# In[5]:


#how many unique classes do we have for categotical features
app_train.select_dtypes('object').apply(pd.Series.nunique, axis = 0)


# In[6]:


#list coloumns with most missing values
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns

missing_values = missing_values_table(app_train)
missing_values.head(30)


# ## Visualizing Missing Data

# In[7]:


import missingno as msno
get_ipython().run_line_magic('matplotlib', 'inline')
msno.matrix(app_train.sample(250))


# In[8]:


msno.bar(app_train.sample(1000))


# In[9]:


#heatmap
msno.heatmap(app_train)


# In[10]:


#Dendogram
msno.dendrogram(app_train)


# In[11]:


# Fixing incorrect data in Days Employed

# Create an anomalous flag column
app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243

# Replace the anomalous values with nan
app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

app_test['DAYS_EMPLOYED_ANOM'] = app_test["DAYS_EMPLOYED"] == 365243
app_test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)


# In[12]:


# Create a label encoder object
le = LabelEncoder()
le_count = 0

# Iterate through the columns
for col in app_train:
    if app_train[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(app_train[col].unique())) <= 2:
            # Train on the training data
            le.fit(app_train[col])
            # Transform both training and testing data
            app_train[col] = le.transform(app_train[col])
            app_test[col] = le.transform(app_test[col])
            
            # Keep track of how many columns were label encoded
            le_count += 1
            
print('%d columns were label encoded.' % le_count)


# In[13]:


# one-hot encoding of categorical variables
app_train = pd.get_dummies(app_train)
app_test = pd.get_dummies(app_test)

print('Training Features shape: ', app_train.shape)
print('Testing Features shape: ', app_test.shape)


# In[14]:


train_labels = app_train['TARGET']

# Align the training and testing data, keep only columns present in both dataframes
app_train, app_test = app_train.align(app_test, join = 'inner', axis = 1)

# Add the target back in
app_train['TARGET'] = train_labels

print('Training Features shape: ', app_train.shape)
print('Testing Features shape: ', app_test.shape)


# In[15]:



# Imputing the data by using the median of each feature to replace relative data
app_train[app_train.columns.tolist()] = SimpleImputer(strategy='median').fit_transform(app_train[app_train.columns.tolist()])
app_test[app_test.columns.tolist()] = SimpleImputer(strategy='median').fit_transform(app_test[app_test.columns.tolist()])


# In[16]:


app_test.info()


# # Feature Selection

# In[17]:


#stratified sampling 
application_sample1 = app_train.loc[app_train.TARGET==1].sample(frac=0.1, replace=False)
print('label 1 sample size:', str(application_sample1.shape[0]))
application_sample0 = app_train.loc[app_train.TARGET==0].sample(frac=0.1, replace=False)
print('label 0 sample size:', str(application_sample0.shape[0]))
application = pd.concat([application_sample1, application_sample0], axis=0).sort_values('SK_ID_CURR')


# In[18]:


application.shape


# In[19]:


X = application.drop(['SK_ID_CURR', 'TARGET'], axis=1)
y = application.TARGET
feature_name = X.columns.tolist()


# ## Filter

# ### Pearson Correlation

# In[20]:


def pcorr(data):
    corrdf = pd.DataFrame(data[data.columns[1:]].corr()['TARGET'][:])
    corrdf = corrdf.drop('TARGET')
    corrdf['TARGET'] = np.abs(corrdf['TARGET'])
    corrdf_graph = corrdf.sort_values('TARGET', ascending = False)
    return corrdf_graph[:100].plot.barh()
pcorr(application)


# In[21]:


def cor_selector(X, y):
    cor_list = []
    # calculate the correlation with y for each feature
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    # replace NaN with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    # feature name
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-75:]].columns.tolist()
    # feature selection? 0 for not select, 1 for select
    cor_support = [True if i in cor_feature else False for i in feature_name]
    return cor_support, cor_feature
cor_support, cor_feature = cor_selector(X, y)


# In[22]:


cor_list = []
for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)


# In[23]:


cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-75:]].columns.tolist()


# ## Wrapper

# ### Recursive Feature Elimination 

# In[24]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
X_norm = MinMaxScaler().fit_transform(X)
rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=75, step=5, verbose=5)
rfe_selector.fit(X_norm, y)


# In[25]:


rfe_support = rfe_selector.get_support()
rfe_feature = X.loc[:,rfe_support].columns.tolist()
ranking = rfe_selector.ranking_

rfe_lists = {'Feature Name': feature_name, 'Rank':ranking}
rfe_total = pd.DataFrame(rfe_lists)
rfe_graph = rfe_total.sort_values('Rank', ascending = True)
rfe_graph.plot.barh()


# ## Embedded

# ### Random Forrest
# 

# In[26]:



from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=75), threshold='1.25*median')
embeded_rf_selector.fit(X, y)

embeded_rf_support = embeded_rf_selector.get_support()
embeded_rf_feature = X.loc[:,embeded_rf_support].columns.tolist()
print(str(len(embeded_rf_feature)), 'selected features')


# In[27]:


pd.set_option('display.max_rows', None)
# put all selection together
feature_selection_df = pd.DataFrame({'Feature':feature_name, 'Pearson':cor_support, 'RFE':rfe_support, 
                                    'Random Forest':embeded_rf_support})
# count the selected times for each feature
feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
# display the top 100
feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
feature_selection_df.index = range(1, len(feature_selection_df)+1)
feature_selection_df.head(75)


# In[28]:


feature_names = feature_selection_df['Feature'].head(75)
feature_name_list = []
for i in range(1,76):
    cor = feature_names[i]
    feature_name_list.append(cor)
feature_name_list


# In[29]:


app_train_feature = app_train[feature_name_list]
app_train_feature["TARGET"]=app_train["TARGET"]


# In[30]:


app_train_feature.head(5)


# ## Model 

# ### Balancing the data using SMOTE

# In[31]:


X = app_train_feature.drop(['TARGET'], axis=1)
y = app_train_feature.TARGET
feature_name = X.columns.tolist()


# In[32]:


pip install imbalanced-learn --user


# In[33]:


import imblearn

import sklearn
from sklearn import datasets
from sklearn.datasets import make_classification
import collections
from collections import Counter
from numpy import where
from matplotlib import pyplot

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# In[34]:


counter = Counter(y)
print(counter)


# In[35]:


over = SMOTE(sampling_strategy=0.1)
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)


# In[36]:


X, y = pipeline.fit_resample(X, y)


# In[37]:


counter = Counter(y)
print(counter)


# ### Logistic Regression

# In[38]:


app_train_feature = X
app_train_feature['TARGET'] = y
app_train_feature.shape


# In[39]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(app_train_feature, test_size=0.2)
print(train.shape)
print(test.shape)


# In[40]:


train_label = train['TARGET']
train = train.drop(['TARGET'], axis=1)
#test_label = test['TARGET']
#test = test.drop(['TARGET'], axis=1)
print(train.shape)
print(train_label.shape)


# In[41]:


log_reg = LogisticRegression(C = 0.0001)
log_reg.fit(train, train_label)


# In[42]:


test_label = test['TARGET']
test = test.drop(['TARGET'], axis=1)
print(test.shape)
print(test_label.shape)


# In[43]:


log_reg_pred = log_reg.predict_proba(test)[:, 1]


# In[44]:


log_reg_pred.shape


# In[45]:


print("AUC Score:", roc_auc_score(test_label, log_reg_pred))


# In[ ]:




