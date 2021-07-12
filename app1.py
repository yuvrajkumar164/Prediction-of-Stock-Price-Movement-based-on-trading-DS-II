#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd    #importing necessary libraries
import numpy as np
import pickle
from joblib import dump, load


# In[2]:


data = pd.read_csv('DataFrame.csv')


# In[3]:


data.drop('Unnamed: 7' , axis = 1 , inplace = True) 


# In[4]:


data.rename(columns = str.upper , inplace = True)


# In[5]:


import datetime
data['DATE'] = pd.to_datetime(data['DATE'] , format = "%Y%m%d")


# In[6]:


#data.head()


# In[7]:


d1 = data.copy()
#d1.head()


# In[8]:


data.drop(['TYPE'],axis=1,inplace=True)


# In[9]:


#data.head()


# In[10]:


X=data.loc[:,['DATE','TIME','OPEN']]


# In[11]:


#X.head()


# In[12]:


X['HOUR']=pd.to_datetime(X['TIME']).dt.hour
X['MINUTE']=pd.to_datetime(X['TIME']).dt.minute
X['DAY']=pd.to_datetime(X.DATE, format="%Y/%m/%d").dt.day
X['MONTH']=pd.to_datetime(X['DATE'], format='%Y/%m/%d').dt.month
X['YEAR']=pd.to_datetime(X['DATE'], format='%Y/%m/%d').dt.year
X.drop(['TIME'],axis=1,inplace=True)
X.drop(['DATE'],axis=1,inplace=True)


# In[13]:


#X.head()


# In[14]:


#X['YEAR'].describe()


# In[15]:


y=data['CLOSE']


# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[18]:


#print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)


# In[19]:


from sklearn.preprocessing import StandardScaler


# In[20]:


sc = StandardScaler()


# In[21]:


sc.fit(X_train)


# In[22]:


X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)
#X_train_std


# In[23]:


filename="stock_dataframe_sc.pkl"
fileobj=open(filename,'wb')
pickle.dump(sc,fileobj)
fileobj.close()


# In[ ]:





# In[ ]:





# In[24]:


from sklearn.svm import SVR
svr1 = SVR(kernel='linear', C = 100)
svr2=SVR(kernel='poly',degree=3,C=20,coef0=1)


# In[ ]:





# In[25]:


svr1.fit(X_train_std,y_train)


# In[26]:


svr2.fit(X_train_std,y_train)


# In[27]:


filename="stock_dataframe_linear.pkl"
fileobj=open(filename,'wb')
pickle.dump(svr1,fileobj)
fileobj.close()


# In[28]:


filename="stock_dataframe_poly.pkl"
fileobj=open(filename,'wb')
pickle.dump(svr2,fileobj)
fileobj.close()


# # MSFT_DATA

# In[29]:


df2 = pd.read_csv('MSFT.csv')
#df2.head()


# In[30]:


df2.rename(columns = str.upper , inplace = True)


# In[31]:


df2['DATE'] = pd.to_datetime(df2['DATE'] , format = "%Y-%m-%d")


# In[32]:


#df2.head()


# In[33]:


d2 = df2.copy()
#d2


# In[34]:


iqr1 = d2['OPEN'].quantile(0.75) - d2['OPEN'].quantile(0.25) 


# In[35]:


upper_whisker = d2['OPEN'].quantile(0.75)+(iqr1*1.5)
lower_whisker = d2['OPEN'].quantile(0.25)-(iqr1*1.5)


# In[36]:


iqr2 = d2['CLOSE'].quantile(0.75) - d2['CLOSE'].quantile(0.25) 


# In[37]:


upper_whisker = d2['CLOSE'].quantile(0.75)+(iqr2*1.5)
lower_whisker = d2['CLOSE'].quantile(0.25)-(iqr2*1.5)


# In[38]:


iqr3 = d2['HIGH'].quantile(0.75) - d2['HIGH'].quantile(0.25) 


# In[39]:


upper_whisker = d2['HIGH'].quantile(0.75)+(iqr3*1.5)
lower_whisker = d2['HIGH'].quantile(0.25)-(iqr3*1.5)


# In[40]:


iqr5 = d2['ADJ CLOSE'].quantile(0.75) - d2['ADJ CLOSE'].quantile(0.25) 


# In[41]:


upper_whisker = d2['ADJ CLOSE'].quantile(0.75)+(iqr5*1.5)
lower_whisker = d2['ADJ CLOSE'].quantile(0.25)-(iqr5*1.5)


# In[42]:


iqr6 = d2['VOLUME'].quantile(0.75) - d2['VOLUME'].quantile(0.25) 


# In[43]:


upper_whisker = d2['VOLUME'].quantile(0.75)+(iqr6*1.5)
lower_whisker = d2['VOLUME'].quantile(0.25)-(iqr6*1.5)


# In[44]:


d2.loc[d2['OPEN'] >= 86.05937385559082 , 'OPEN'] = 86.05937385559082
d2.loc[d2['CLOSE'] >= 85.87265515327454 , 'CLOSE'] = 85.87265515327454
d2.loc[d2['HIGH'] >= 86.6875 , 'HIGH'] = 86.6875
d2.loc[d2['LOW'] >= 84.5562515258789 , 'LOW'] = 84.5562515258789
d2.loc[d2['ADJ CLOSE'] >= 64.37626564502716 , 'ADJ CLOSE'] = 64.37626564502716
d2.loc[d2['VOLUME'] >= 130092200.0 , 'VOLUME'] = 130092200.0


# In[45]:


#d2.head()


# In[46]:


d2['DAY']=pd.to_datetime(d2.DATE, format="%Y/%m/%d").dt.day

d2['MONTH']=pd.to_datetime(d2['DATE'], format='%Y/%m/%d').dt.month

d2['YEAR']=pd.to_datetime(d2['DATE'], format='%Y/%m/%d').dt.year


# In[47]:


#d2.head()


# In[48]:


d2.drop(['DATE'],axis=1,inplace=True)
#d2['YEAR'].describe()


# In[49]:


y2=d2['CLOSE']
X2=d2.loc[:,['OPEN','DAY','MONTH','YEAR']]


# In[50]:


X2_train,X2_test,y2_train,y2_test=train_test_split(X2,y2,test_size=0.2,random_state=0)


# In[51]:


sc.fit(X2_train)


# In[52]:


filename="stock_MSFT_sc.pkl"
fileobj=open(filename,'wb')
pickle.dump(sc,fileobj)
fileobj.close()


# In[53]:


X2_train_std=sc.transform(X2_train)


# In[54]:


X2_test_std=sc.transform(X2_test)


# In[55]:


#X2_train_std


# In[56]:


svr3 = SVR(kernel='linear', C = 10)


# In[57]:


#X2_test.head()


# In[58]:


svr3.fit(X2_train_std,y2_train)


# In[59]:


svr4=SVR(kernel='poly',degree=3,C=10,coef0=1)


# In[60]:


svr4.fit(X2_train_std,y2_train)


# In[61]:


filename="stock_MSFT_linear.pkl"
fileobj=open(filename,'wb')
pickle.dump(svr3,fileobj)
fileobj.close()


# In[62]:


filename="stock_MSFT_poly.pkl"
fileobj=open(filename,'wb')
pickle.dump(svr4,fileobj)
fileobj.close()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:



#y_test_pred[0]


# In[ ]:


#y_test_pred=svr1.predict(X_test_std)
#y_test_pred[0]


# In[ ]:


#y_train_pred=svr1.predict(X_train_std)


# In[ ]:


#from sklearn.metrics import mean_squared_error, r2_score#at c=50


# In[ ]:


#mean_squared_error(y_test,y_test_pred)


# In[ ]:


#mean_squared_error(y_train,y_train_pred)


# In[ ]:


#r2_score(y_test,y_test_pred)


# In[ ]:


#r2_score(y_train,y_train_pred)


# # MSFT

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # DEPLOYEMENT

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




