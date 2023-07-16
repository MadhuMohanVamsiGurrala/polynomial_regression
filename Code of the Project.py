#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import numpy as np    #version 1.21.5
import pandas as pd   #version  1.4.4


# In[2]:


#importing dataset
dataset = pd.read_csv(r'E:\Python\50_Startups.csv')
dataset.head()


# In[3]:


#assingning values to x and y
X=dataset.iloc[:,0:2].values
y=dataset.iloc[:,-1].values


# In[4]:


#diving the dataset into training and testing data
from sklearn.model_selection import train_test_split  #version 0.0.post1
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)


# In[5]:


#Create polynomial features up to degree 2
from sklearn.preprocessing import PolynomialFeatures 
poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)


# In[6]:


#Train a linear regression model on the polynomial features
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(X_train_poly, y_train)
y_predict=regressor.predict(X_test_poly)


# In[7]:


#Mean Absolute Error
from sklearn import metrics
print(metrics.mean_absolute_error(y_test,y_predict))


# In[8]:


#Mean Squared Error
print(metrics.mean_squared_error(y_test,y_predict))


# In[9]:


#Root Mean Squared Error
print(np.sqrt(metrics.mean_squared_error(y_test,y_predict)))


# In[ ]:




