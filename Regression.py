#!/usr/bin/env python
# coding: utf-8

# __Regression__ <br>

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random


# -Read the data set “Life Expectancy Data.csv”.
# 
# This dataset provides the life expectancy of people under various metrics (independent variables) of the World Health Organization. 'Life expectancy' is the dependent variable and the rest are independent variables.
# 
# https://gist.github.com/aishwarya8615/89d9f36fc014dea62487f7347864d16a

# In[4]:


led = pd.read_csv("Life Expectancy Data.csv")
df = led.copy()
df.head(10)


# In[5]:


df = df.drop(['Country', 'Year', 'Status'], axis=1)


# In[6]:


df


# In[7]:


observation_unit=len(df)
variable_num= len(df.columns)
print("observation unit:",observation_unit,"variable number:",variable_num)


# In[8]:


variable_name= df.columns
variable_name


# In[9]:


independent_variable= ['Adult Mortality', 'infant deaths', 'Alcohol', 'percentage expenditure', 'Hepatitis B', 'Measles ', ' BMI ', 'under-five deaths ', 'Polio''Total expenditure','Diphtheria','HIV/AIDS','GDP','Population','thinness  1-19 years','thinness 5-9 years','Income composition of resources','Schooling']
independent_variable


# In[10]:


missing_values= df.isnull().sum()
missing_values


# In[11]:


df = df.fillna(df.mean())


# In[12]:


df


# In[13]:


df.isnull().sum()


# In[14]:


X = df.drop(['Life expectancy'], axis=1)

y= df['Life expectancy']

print("Independent Variables (X):")
print(X)

print("\nDependent Variables (y):")
print(y)


# In[15]:


from sklearn.model_selection import train_test_split, cross_val_score


# In[16]:


X = df.drop(['Life expectancy'], axis=1)

y = df['Life expectancy']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state= 42)


# 14- LinearRegression kütüphanesini yükleyin
# __model__ oluşturup fit edin

# In[17]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(X_train, y_train)


# In[18]:


model.intercept_


# In[19]:


model.coef_


# In[20]:


from sklearn.metrics import mean_squared_error, r2_score


# In[21]:


y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Square Error (MSE):", mse)

r2 = r2_score(y_test, y_pred)
print("R-square (R2 Score):", r2)


# In[22]:


rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
rmse


# 19- Tüm Test verisi üzerinde modelin Test hatasını gösterin - root mean square

# In[23]:


rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
rmse


# In[24]:


from sklearn.metrics import r2_score
y_train_pred = model.predict(X_train)
r2_train = r2_score(y_train, y_train_pred)
r2_train


# In[25]:


from sklearn.metrics import r2_score
y_test_pred = model.predict(X_test)
r2_test = r2_score(y_test, y_test_pred)
r2_test


# In[26]:


from sklearn.model_selection import cross_val_score


# In[27]:


cross_val_score(model, X_train, y_train, cv = 10, scoring = "r2").mean()


# In[28]:


np.sqrt(-cross_val_score(model, 
                X_test, 
                y_test, 
                cv = 10, 
                scoring = "neg_mean_squared_error")).mean()


# ##### PCR Model

# In[29]:


from sklearn.decomposition import PCA 
from sklearn.preprocessing import scale


# In[30]:


pca = PCA()


# In[31]:


X_reduced_train = pca.fit_transform(scale(X_train)) 


# In[32]:


np.round(pca.explained_variance_ratio_.cumsum()*100, decimals=2)


# In[33]:


import matplotlib.pyplot as plt
plt.figure(figsize=(9,3))
plt.axhline(90, c="r")
plt.plot(pca.explained_variance_ratio_.cumsum()*100);


# Answer: As can be seen from the graph, when we take about 8 (components) components, we get to a level where we can represent 90% of the total data. As a result, it is observed that as the number of components increases, the rate of explaining the variance in the original data set increases.

# In[34]:


from sklearn.linear_model import LinearRegression
lm_pca = LinearRegression()
pcr_model = lm_pca.fit(X_reduced_train, y_train)


# In[35]:


#b0
pcr_model.intercept_


# In[36]:


pcr_model.coef_


# In[37]:


y_pred_pcr_train = pcr_model.predict(X_reduced_train)
y_pred_pcr_train[0:10]


# In[38]:


np.sqrt(mean_squared_error(y_train, y_pred_train))


# In[39]:


lm = LinearRegression()
model = lm.fit(X_test, y_test)
y_pred_test = model.predict(X_test)

X_reduced_test = pca.fit_transform(scale(X_test)) 
y_pred_pcr_test = pcr_model.predict(X_reduced_test)

hata_t = pd.DataFrame({"real_y":     y_test[0:10],
                       "prediction_y":     y_pred_test[0:10],
                       "prediction_y_pcr": y_pred_pcr_test[0:10],})


# In[40]:


error_t

