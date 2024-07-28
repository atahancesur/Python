#!/usr/bin/env python
# coding: utf-8

# __Statistics for Data Science__ <br> 

# -Let it be argued that the average commuting time of people working in Istanbul is 90 minutes.

# In[1]:


import numpy as np 
minutes = np.array([45,48,96,28,76,36,60,61,69,58,78,67,69,72,22,68,39,10,18,110,60,60,120,45,96,78,32,
                      68,78,61,69,94,102,105,103,100,75,60,60,36,61,23,88,89,88,65,27,36,120,60])


# Hypotheses 
# * H0 : The average commuting time of people working in Istanbul is 90 minutes.
# * H1 : The average commuting time of people working in Istanbul is not 90 minutes. 

# In[2]:


import scipy.stats as stats
import pandas as pd 
import seaborn as sns


# -Convert np array to DataFrame.

# In[3]:


df = pd.DataFrame(minutes, columns=['time'])
df.head(5)


# -Show descriptive statistics via DataFrame

# In[4]:


df.describe().T


# In[5]:


pd.DataFrame(minutes).plot.hist();


# Comment: This graph shows that the commute time of people working in Istanbul is between 60-80 minutes.

# Hypotheses Hypothesis H0: There is no difference between the sample distribution and the normal distribution
#             In Hypothesis H1, there is a difference between the sample distribution and the normal distribution
#             Objective: To reject Hypothesis H0.

# In[6]:


from scipy.stats import shapiro
shapiro(minutes)


# Comment: According to this test, <br> H0 is not rejected <br> since the p_value is greater than alpha = 0.05.
# In other words, the sample is normally distributed and parametric One sample T test can be used.

# In[7]:


stats.ttest_1samp(minutes, popmean = 90)


# Since our test value, p_value, is greater than 0.05, our hypothesis is rejected. In this case, the average time spent on the road in Istanbul is 90 minutes. As a result, the hypothesis is rejected.

# -Apply the one-sample T-test, which is a non-parametric test.

# In[8]:


from statsmodels.stats.descriptivestats import sign_test
sign_test(minutes, 90)


# Comment: We are not interested in the first value of -15. We are interested in the pvalue. pvalue =8.056738293263073e-08 and since it is not less than 0.05, we cannot reject hypothesis H0. As a result, this test gave us a misleading result.

# In[9]:


F = pd.DataFrame([75,96,26,41,98,35,74,55,69,85,88,89,45,46,17,13,14,84,96,90,10,10,78,65,64,86,85,98,25,35,21])
M = pd.DataFrame([74,80,20,35,90,30,71,52,43,45,42,41,75,87,65,99,100,27,15,48,68,24,24,45,78,88,99,99,100,35,6])


# -My hypotheses.

# <d><i>H</i><sub>0</sub>: Women's physics test scores are equal to men's physics test scores
#     
# <d><i>H</i><sub>1</sub>: Women's physics test scores are not equal to men's physics test scores.

# -Combined male and female data by column.

# In[10]:


F_M = pd.concat([F, M], axis = 1)
F_M.columns = ["F","M"]

F_M.head()


# -Two separate data categories merged into a DataFrame.

# In[11]:


GROUP_F = np.arange(len(F))
GROUP_F = pd.DataFrame(GROUP_F)
GROUP_F[:] = "K"
F = pd.concat([F, GROUP_F], axis = 1)  

GROUP_M = np.arange(len(M))
GROUP_M = pd.DataFrame(GROUP_M)
GROUP_M[:] = "M"
M = pd.concat([M, GROUP_M], axis = 1)


FM = pd.concat([F,M], axis = 0)
FM.columns = ["Score","GROUP"]
print(FM.head())
print(FM.tail())


# -Boxplot plotted with seaborn library using data type II

# In[12]:


import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


# In[13]:


sns.boxplot(x = "GROUP", y = "Score", data = FM);


# Comment: As far as we can see in the graph, the average score of women in the physics test is higher than that of men. We need to do a few more tests to confirm this.

# * H0 : There is no statistical difference between the sample distribution and the normal distribution.
# * H1 : There is a statistical difference between the sample distribution and the normal distribution.

# In[14]:


females = pd.DataFrame([75,96,26,41,98,35,74,55,69,85,88,89,45,46,17,13,14,84,96,90,10,10,78,65,64,86,85,98,25,35,21])
males = pd.DataFrame([74,80,20,35,90,30,71,52,43,45,42,41,75,87,65,99,100,27,15,48,68,24,24,45,78,88,99,99,100,35,6])


# In[15]:


shapiro(females)


# In[16]:


shapiro(males)


# Comment: According to this test, we cannot reject hypothesis H0 because our pvalue is greater than 0.05.

# -We test the assumption of variance homeogeneity.

# In[17]:


stats.levene(F_M.F, F_M.M)


# Comment: Since the pvalue is greater than 0.05, we cannot reject hypothesis H0.

# Independent Two Sample T Test (Hypothesis Testing Parametric Test)

# In[18]:


from scipy.stats import shapiro


# In[19]:


stats.ttest_ind(F_M["F"], F_M["M"], equal_var = True)


# In[20]:


test_statistics, pvalue = stats.ttest_ind(F_M["F"], F_M["M"], equal_var = True)
print('Test Statistics = %.4f, p-value = %.4f' % (test_statistics, pvalue))


# Comment: F = M is accepted. pvalue greater than 0.05.

# Nonparametric Independent Two-Sample Test

# In[21]:


stats.mannwhitneyu(F_M["F"], F_M["M"])


# In[22]:


test_statistics, pvalue = stats.mannwhitneyu(F_M["F"], F_M["M"])
print('est Statistics = %.4f, p-value = %.4f' % (test_statistics, pvalue))


# Comment: F = M is accepted. pvalue greater than 0.05.

# 
