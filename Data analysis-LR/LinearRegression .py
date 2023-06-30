#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


# In[2]:


data = pd.read_csv('reallifedata.csv')


# In[3]:


data


# In[4]:


#preprocesssing


# In[5]:


data.describe()


# In[6]:


data.describe(include='all')


# In[7]:


data = data.drop(['Model'],axis=1)


# In[8]:


data


# In[9]:


# Missing Values ----> Null values


# In[10]:


data.describe(include='all')


# In[11]:


data.isnull()


# In[12]:


data.isnull().sum()


# In[13]:


data


# In[14]:


data_no_mv= data.dropna(axis=0)


# In[15]:


data_no_mv.describe(include='all')


# In[16]:


data_no_mv.isnull().sum()


# In[17]:


# Normallity & Outlier


# In[18]:


sns.distplot(data_no_mv['Price'])
plt.show()


# In[19]:


q = data_no_mv['Price'].quantile(0.99)


# In[20]:


q


# In[21]:


data_1 = data_no_mv[data_no_mv['Price']<q]


# In[22]:


data_1


# In[23]:


sns.distplot(data_1['Price'])
plt.show()


# In[24]:


sns.distplot(data_no_mv['Mileage'])
plt.show()


# In[25]:


q = data_1['Mileage'].quantile(0.99)


# In[26]:


q


# In[27]:


data_2 = data_1[data_1['Mileage']<q]


# In[28]:


data_2


# In[29]:


sns.distplot(data_2['Mileage'])
plt.show()


# In[30]:


sns.distplot(data_no_mv['EngineV'])
plt.show()


# In[31]:


data_3 = data_2[data_2['EngineV']<6.5]


# In[32]:


data_3


# In[33]:


sns.distplot(data_3['EngineV'])
plt.show()


# In[34]:


sns.distplot(data_no_mv['Year'])
plt.show()


# In[35]:


q = data_3['Year'].quantile(0.01)


# In[36]:


q


# In[37]:


data_4 = data_3[data_3['Year']>q]


# In[38]:


sns.distplot(data_4['Year'])
plt.show()


# In[39]:


# Linearity


# In[40]:


data_4


# In[41]:


plt.scatter(data_4['Mileage'],data_4['Price'])
plt.xlabel('Mileage')
plt.ylabel('Price')
plt.show()


# In[42]:


plt.scatter(data_4['EngineV'],data_4['Price'])
plt.xlabel('EngineV')
plt.ylabel('Price')
plt.show()


# In[43]:


plt.scatter(data_4['Year'],data_4['Price'])
plt.xlabel('Year')
plt.ylabel('Price')
plt.show()


# In[44]:


f,(ax1,ax2,ax3) = plt.subplots(1,3,sharey=True,figsize=(15,3))
ax1.scatter(data_4['Mileage'],data_4['Price'])
ax1.set_title('Mileage vs Price')
ax2.scatter(data_4['EngineV'],data_4['Price'])
ax2.set_title('EngineV vs Price')
ax3.scatter(data_4['Year'],data_4['Price'])
ax3.set_title('Year vs Price')
plt.show()


# In[45]:


data_4


# In[46]:


data_cleaned = data_4.reset_index(drop=True)


# In[47]:


data_cleaned


# In[48]:


sns.distplot(data_cleaned['Price'])
plt.show()


# In[49]:


# faceing Exponential Dist^ -----> Solution ---------> Log Transformation


# In[50]:


data_cleaned['Price']


# In[51]:


log_price = np.log(data_cleaned['Price'])


# In[52]:


log_price


# In[53]:


sns.distplot(log_price)
plt.show()


# In[54]:


np.exp(log_price)


# In[55]:


f,(ax1,ax2,ax3) = plt.subplots(1,3,sharey=True,figsize=(15,3))
ax1.scatter(data_4['Mileage'],log_price)
ax1.set_title('Mileage vs log_Price')
ax2.scatter(data_4['EngineV'],log_price)
ax2.set_title('EngineV vs log_Price')
ax3.scatter(data_4['Year'],log_price)
ax3.set_title('Year vs log_Price')
plt.show()


# In[56]:


# multicolinearity


# In[57]:


multico = data_cleaned[['Mileage','EngineV','Year']]


# In[58]:


multico


# In[59]:


multico.corr()


# In[60]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[61]:


vif = [variance_inflation_factor(multico.values,i) for i in range(3)]


# In[62]:


vif


# In[63]:


data_cleaned = data_cleaned.drop(['Year'],axis=1)


# In[64]:


data_cleaned


# In[65]:


data_cleaned = data_cleaned.drop(['Price'],axis=1)


# In[66]:


data_cleaned['log_price'] = log_price


# In[67]:


data_cleaned


# In[68]:


data_cleaned.Brand.unique()


# In[69]:


data_cleaned.Registration.unique()


# In[70]:


data_cleaned.describe(include='all')


# In[71]:


data_cleaned


# In[72]:


data_with_dummies = pd.get_dummies(data_cleaned,drop_first=True)


# In[73]:


data_with_dummies


# In[74]:


X = data_with_dummies.drop(['log_price'],axis=1)


# In[75]:


X


# In[76]:


y = data_with_dummies['log_price']


# In[77]:


y


# In[78]:


sns.distplot(X['Mileage'])
plt.show()


# In[79]:


from sklearn.preprocessing import StandardScaler


# In[80]:


scaler = StandardScaler()


# In[81]:


X_scaled = scaler.fit_transform(X)


# In[82]:


X_scaled


# In[83]:


X_scaled.shape


# In[84]:


test_df = pd.DataFrame(X_scaled)


# In[85]:


test_df


# In[86]:


sns.distplot(test_df[0])


# In[87]:


X_scaled


# In[88]:


y


# In[89]:


from sklearn.model_selection import train_test_split


# In[90]:


X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.2,random_state=0)


# In[91]:


3867*0.8


# In[92]:


X_train.shape


# In[93]:


X_test.shape


# In[94]:


from sklearn.linear_model import LinearRegression


# In[95]:


model = LinearRegression()


# In[96]:


model.fit(X_train,y_train)


# In[97]:


model.score(X_test,y_test)


# In[98]:


y_pred = model.predict(X_test)


# In[99]:


y_pred


# In[100]:


y_test


# In[101]:


plt.scatter(y_test,y_pred)
plt.xlabel('Actual Data')
plt.ylabel('Predictions')
plt.show()


# In[102]:


y_pred


# In[103]:


y_pred = np.exp(y_pred)


# In[104]:


y_pred


# In[ ]:




