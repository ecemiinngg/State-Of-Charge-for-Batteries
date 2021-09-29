#!/usr/bin/env python
# coding: utf-8

# In[146]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

from sklearn.linear_model import Lasso , Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor , GradientBoostingRegressor 
from sklearn.neural_network import MLPRegressor 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

import warnings 
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore',category=ConvergenceWarning)
from sklearn.exceptions import NotFittedError
import pickle
df = pd.read_csv("C:\\Users\\ecemi\\Desktop\\data.csv",sep=";")
df.info()
df.head()


# In[150]:


df.hist(figsize=(14,14),xrot=-45)
plt.show


# In[128]:


####!!!sütun isimlerinde whitespace var 
df = df.rename(columns=lambda col: col.strip())


# In[130]:


df.SOC.hist()
plt.show()


# In[134]:


correlations = df.corr()
correlations


# In[136]:


plt.figure(figsize=(7,6))
sns.heatmap(correlations,cmap='RdBu_r')
plt.show()


# In[138]:


###data temizliği
print(df.shape)
df = df.drop_duplicates()
print(df.shape)


# In[140]:


### bütün sütunları integer'a çevirdik
for col in df.columns :
    try:
        df[col]=pd.to_numeric(df[col].str.replace(',','.'),errors='coerce')
     # veri tipi str değilse hatayı görmezden gel
    except AttributeError:
        continue


# In[142]:


###feature engineering
Vavg_num = 100
V = []
V[:Vavg_num] = df.voltage[:Vavg_num]
for i in range (Vavg_num,len(df.voltage)):
    V.append(np.mean(df.voltage[i-Vavg_num:i]))
df['Vmean']= V
df.tail()


# In[144]:


####algoritma
y= df['SOC']
X= df.drop('SOC',axis=1)


# In[158]:


X_train, X_test , y_train, y_test =train_test_split(X,y,test_size=0.2 , random_state=1234)

pipelines = {
    'lasso':make_pipeline (StandardScaler(),Lasso(random_state=123)),
    'ridge':make_pipeline(StandardScaler(),Ridge(random_state=123)),
    'enet':make_pipeline(StandardScaler(),ElasticNet(random_state=123)),
    'rf':make_pipeline(StandardScaler(),RandomForestRegressor(random_state=123)),
    'gb':make_pipeline(StandardScaler(),GradientBoostingRegressor(random_state=123)),
    'mlp':make_pipeline(StandardScaler(),MLPRegressor(random_state=123))
        
}
for key, value in pipelines.items():
    print( key, type(value) )


# In[159]:


MLPRegressor()


# In[160]:


lasso_hyperparameters = {
    'lasso__alpha':[0.0001, 0.001,0.01,1,5,10]
    }

ridge_hyperparameters = {
    'ridge__alpha':[0.0001,0.001,0.01,0.1,1,5,10]
    }
enet_hyperparameters = {
    'elasticnet__alpha':[0.0001,0.001,0.01,0.1,1,5,10],
    'elasticnet__l1_ratio':[0.1,0.3,0.5,0.7,0.9]
        }

rf_hyperparameters = {
    'randomforestregressor__n_estimators':[10,20],
    'randomforestregressor__max_features':['auto','sqrt',0.33]
    }


gb_hyperparameters = {
    'gradientboostingregressor__n_estimators':[10,20],
    'gradientboostingregressor__learning_rate':[0.05,0.1,0.2],
    'gradientboostingregressor__max_depth':[1,3,5]
    }
mlp_hyperparameters = {
    'mlpregressor__hidden_layer_sizes':[(100,)],
    'mlpregressor__activation':['logistic','relu']
    
    } 

hyperparameters ={
    'lasso': lasso_hyperparameters,
    'ridge':ridge_hyperparameters,
    'enet':enet_hyperparameters,
    'rf':rf_hyperparameters,
    'gb':gb_hyperparameters,
    'mlp':mlp_hyperparameters
    }


# In[161]:


for key in ['enet', 'gb', 'ridge', 'rf', 'lasso', 'mlp']:
    if key in hyperparameters:
        if type(hyperparameters[key]) is dict:
            print( key, 'was found in hyperparameters, and it is a grid.' )
        else:
            print( key, 'was found in hyperparameters, but it is not a grid.' )
    else:
        print( key, 'was not found in hyperparameters')


# In[162]:


## boş dict
fitted_models = {}


# In[163]:


X_train = X_train.fillna(X_train.mean())
for name, pipeline in pipelines.items ():
    model= GridSearchCV(pipeline,hyperparameters[name],cv=10,n_jobs=-1)
    model.fit(X_train,y_train)
    fitted_models[name] = model
    print(name,"has been fitted")
    
    


# In[167]:


X_test = X_test.fillna(X_test.mean())
for name, model in fitted_models.items():
    try:
        pred = model.predict(X_test)
        print(name,"has been fitted")
    except NotFittedError as e :
        print(repr(e))


# In[168]:


###model selection
for name, model in fitted_models.items():
    print(name,model.best_score_)


# In[169]:


for name, model in fitted_models.items(): 
    pred = fitted_models[name].predict(X_test)    
    print(name)
    print('R2:',r2_score(y_test,pred))
    print('MAE:',mean_absolute_error(y_test, pred))


# In[170]:


#### RANDOM FOREST WINSSSS


# In[171]:


rf_pred = fitted_models["rf"].predict(X_test)
plt.scatter(rf_pred , y_test)
#tahmin edilen değerler
plt.xlabel('predicted') 
#gerçek değerler
plt.ylabel('actual')
plt.show()


# In[ ]:




