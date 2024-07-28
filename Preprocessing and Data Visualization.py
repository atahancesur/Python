#!/usr/bin/env python
# coding: utf-8

# __Preprocessing and Data Visualization__ <br>

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random


# In[2]:


import pandas as pd
df = pd.read_csv('Life Expectancy Data.csv')
df.head(10)


# -Remove the first three variables from the data set (Country, Year, Status)

# In[3]:


df = df.drop(['Country', 'Year', 'Status'], axis=1)
df.head(5)


# -Show the descriptive statistics of the data set

# In[4]:


df.describe().T


# -Show the number of observation units and the number of variables in the data set

# In[5]:


observation_unit=len(df)
variable_num= len(df.columns)
print("observation unit:",observation_unit,"variable number:",variable_num)


# -Show the names of variables

# In[6]:


variable_name = df.columns
variable_name


# -Discard all columns of the dataset after 'Polio' from Polio ('Polio' should be left in the Dataset) <br>
# Note: This will be the dataset we will consider in the following processes.

# In[7]:


polio_index = df.columns.get_loc('Polio')
df_previous_column = df.iloc[:, :polio_index]
df = df.loc[:, :'Polio']
df


# -Display the names of the arguments

# In[8]:


independent_variable= ['Adult Mortality', 'infant deaths', 'Alcohol', 'percentage expenditure', 'Hepatitis B', 'Measles ', ' BMI ', 'under-five deaths ', 'Polio']
independent_variable


# -Show the number of missing values in each variable (including both dependent and independent variables)

# In[9]:


missing_values = df.isnull().sum()
missing_values


# -Visualize Missing Data

# In[10]:


import missingno as msno
msno.matrix(df);


# -Find and comment on the highest missing correlation by removing the dependent variable

# In[11]:


msno.heatmap(df);


# Comment: Using a heatmap we can see their (nullity correlation) missingness correlation values.
# Supporting the inference above, this graph gives us the pairwise correlation values. The highest nullity correlation is between BMI and Polio.

# -Fill in the missing values by finding the minimum and maximum values column-wise (variable-wise) and generating random values between them.<br>
# Note: Do not corrupt the original data set: Make a copy or save it under another name.<br>
# Set seed to 0 to generate the same random numbers.

# In[12]:


# using loop 
df.fillna(df.mean()[:])


# In[13]:


# or without loop (using apply and lambda)
df.apply(lambda x: x.fillna(x.mean()), axis = 0)


# -After filling in, again show the number of missing values in each variable (including both dependent and independent variables)

# In[14]:


var_names = list(df)
var_names


# In[15]:


from ycimpute.imputer import knnimput
import numpy as np
n_df = np.array(df)


# In[16]:


dff = knnimput.KNN(k = 4).complete(n_df)


# In[17]:


import pandas as pd
dff = pd.DataFrame(dff, columns = var_names)


# In[18]:


dff.isnull().sum()


# -Draw a correlation graph for the variables in the dataset, including the dependent variable (without missing data). Comment on the negative and positive correlation of the independent variable with the dependent variables.

# In[19]:


df_no_missing_values = df.dropna()

correlation_matrix = df_no_missing_values.corr()

plt.figure(figsize=(6.5, 6.5))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".1f", linewidths=.5)
plt.show()


# Comment: Variables in the dataset, including the dependent variable (with no missing data) We can see their values using a correlation graph. Infant mortality and deaths of children under the age of 5 share a common denominator.

# In[ ]:




