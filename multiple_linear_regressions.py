# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 20:52:14 2018

@author: Manan
"""

#Import the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Import the Dataset
dataset=pd.read_csv('50_Startups.csv')

x=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,4].values

#Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
x[:, 3] = labelencoder.fit_transform(x[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()

# Avoiding the Dummy Variable Trap
x = x[:, 1:]

#Splitting the dataset into training set and test set
 
from sklearn.cross_validation import train_test_split
 
X_train,X_test,Y_train,Y_test=train_test_split(x,Y,test_size=0.2,random_state=0)


#Fitting multiple regression model to training set

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
 
#Predicting the test set result
Y_pred=regressor.predict(X_test)

#Building the optimal model using backward elimination
import statsmodels.formula.api as sm
x=np.append(arr=np.ones((50,1)).astype(int),values=x,axis=1)

X_opt=x[:,[0,1,2,3,4,5]]
regressor_OLS= sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()


X_opt=x[:,[0,1,3,4,5]]
regressor_OLS= sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()


X_opt=x[:,[0,3,4,5]]
regressor_OLS= sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()



X_opt=x[:,[0,3,5]]
regressor_OLS= sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()


X_opt=x[:,[0,3]]
regressor_OLS= sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()


from sklearn.cross_validation import train_test_split
X_opt_train,X_opt_test=train_test_split(X_opt,test_size=0.2, random_state=0)


from sklearn.linear_model import LinearRegression
regressor_opt=LinearRegression()
regressor_opt.fit(X_opt_train,Y_train)

Y_opt_pred=regressor_opt.predict(X_opt_test)
