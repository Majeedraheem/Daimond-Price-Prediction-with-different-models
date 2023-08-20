#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 14:08:40 2019

@author: mac
"""

import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt # plotting
import warnings; warnings.filterwarnings("ignore")
from pandas.api.types import CategoricalDtype
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report, mean_squared_error
from sklearn import metrics
from math import sqrt
import seaborn as sns
import os 

os.getcwd()
os.chdir('/Users/mac/Desktop/ML /dataset/')
df= pd.read_csv('Daimonds.csv')
#df_sample=pd.read_csv('Daimonds.csv.csv',nrows=10000, na_values='NA',        )
df.isnull().sum()


#drop unncessery column index 
df.drop(['Unnamed: 0'],axis=1,inplace=True)
#check the dataset statisticalley 
df.describe().T
dfc=df.copy()
cols=df.columns
### Get Dummies for cut, color ,clarity categories features 
df_dum =pd.get_dummies(df,columns=['cut', 'color', 'clarity'],drop_first=True)
###for kflod perpouse 
df_dum_c=df_dum.copy()
x_k=df_dum_c.drop(['price'],axis=1)
y_k=df_dum_c.price
################

df_dum_train,df_dum_test = train_test_split(df_dum, test_size = 0.3, random_state = 0)



#scalling 
sc = StandardScaler()
fdf_scaled_train = pd.DataFrame(sc.fit_transform(df_dum_train), columns=df_dum.columns)
fdf_scaled_test = pd.DataFrame(sc.transform(df_dum_test), columns=df_dum.columns)




# dependent feature Y and independence features X
X_train = fdf_scaled_train.drop(['price'], axis = 1)
Y_train = fdf_scaled_train.price

X_test = fdf_scaled_test.drop(['price'], axis = 1)
Y_test = fdf_scaled_test.price

#####
corr = df.corr()
sns.heatmap(data=corr, square=True , annot=True, cbar=True)
#####
####add constant 
X1=sm.add_constant(X_train) 
ols=sm.OLS(Y_train,X1)
lr=ols.fit()
lr.pvalues
pvalue=max(lr.pvalues[1:len(lr.pvalues)])
print(lr.summary())
#drop P value greater than 0.05
while (pvalue>=0.05):
    loc=0
    for i in lr.pvalues:
        if (i==pvalue):
            feature=lr.pvalues.index[loc]
            print(feature)
        loc+=1
    X_train=X_train.drop(feature,axis=1)
    X_test=X_test.drop(feature,axis=1)
    X1=sm.add_constant(X_train) 
    ols=sm.OLS(Y_train,X1)
    lr=ols.fit()
    pvalue=max(lr.pvalues[1:len(lr.pvalues)])
    print(lr.summary()

############## 
'''
=================================================================================
                    coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------
const         -6.289e-17      0.001   -4.3e-14      1.000      -0.003       0.003
carat             1.3230      0.007    192.385      0.000       1.310       1.337
depth            -0.0248      0.002    -14.090      0.000      -0.028      -0.021
table            -0.0164      0.002     -8.396      0.000      -0.020      -0.013
x                -0.2736      0.007    -39.751      0.000      -0.287      -0.260
cut_Good          0.0391      0.003     13.481      0.000       0.033       0.045
cut_Ideal         0.0977      0.005     19.897      0.000       0.088       0.107
cut_Premium       0.0789      0.004     18.724      0.000       0.071       0.087
cut_Very Good     0.0729      0.004     18.093      0.000       0.065       0.081
color_E          -0.0175      0.002     -8.436      0.000      -0.022      -0.013
color_F          -0.0228      0.002    -10.967      0.000      -0.027      -0.019
color_G          -0.0458      0.002    -21.193      0.000      -0.050      -0.042
color_H          -0.0851      0.002    -41.749      0.000      -0.089      -0.081
color_I          -0.1080      0.002    -56.426      0.000      -0.112      -0.104
color_J          -0.1298      0.002    -74.543      0.000      -0.133      -0.126
clarity_IF        0.2400      0.003     88.155      0.000       0.235       0.245
clarity_SI1       0.3943      0.006     71.221      0.000       0.383       0.405
clarity_SI2       0.2547      0.005     52.189      0.000       0.245       0.264
clarity_VS1       0.4111      0.005     87.213      0.000       0.402       0.420
clarity_VS2       0.4489      0.005     82.472      0.000       0.438       0.460
clarity_VVS1      0.3166      0.004     89.942      0.000       0.310       0.324
clarity_VVS2      0.3623      0.004     91.374      0.000       0.355       0.370
==============================================================================
'''
#check the corr between features 
plt.figure(figsize=(50,50))
sns.heatmap(df.corr(),linewidths=0.1,linecolor='black',square=True,cmap='summer')  
fdf_scaled_test.describe().T
    
############################modeling with linear Regresstion 
from sklearn.model_selection import cross_val_score

model_l=LinearRegression()
model_l.fit(X_train, Y_train)
y_pred = model_l.predict(X_test)

model_l.coef_
model_l.intercept_
RMSE= sqrt(mean_squared_error(Y_test,y_pred))

list_cof=list(zip(cols,model_l.coef_))
df_targets = pd.DataFrame({'Actual': Y_test, 'Predicted': y_pred})
df_targets.describe().T
#RMSE    
#print (np.sqrt(-cross_val_score(model_l,X_train,y,cv=3,scoring='neg_mean_squared_error')).mean())
#testing model using Kfold with 4 splits 
print(RMSE)
print('R-squared: Between Test And Predict',r2_score(Y_test, y_pred))
print('R-squared: Between Train And Predict',r2_score(Y_train,model_l.predict(X_train)))
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))  


'''
print(r2_score(Y_test, y_pred))
0.9209268693223643

print(r2_score(Y_train,model_l.predict(X_train)))
0.9192587702706269
'''
'''
==============================================================================
Dep. Variable:                  price   R-squared:                       0.919
Model:                            OLS   Adj. R-squared:                  0.919
Method:                 Least Squares   F-statistic:                 2.046e+04
Date:                Mon, 04 Nov 2019   Prob (F-statistic):               0.00
Time:                        15:23:39   Log-Likelihood:                -6067.2
No. Observations:               37758   AIC:                         1.218e+04
Df Residuals:                   37736   BIC:                         1.237e+04
Df Model:                          21                                         
Covariance Type:            nonrobust                                         
===============================================================================
''''

########linear Reg modeling 
model = LinearRegression()
scores = []
kfold = KFold(n_splits=4, shuffle=True, random_state=23)
for i, (train, test) in enumerate(kfold.split(x_k,y_k)):
 model.fit(x_k.iloc[train], y_k.iloc[train])
 score = model.score(x_k.iloc[test], y_k.iloc[test])
 scores.append(score)
 s=np.array(scores)
 s.mean()
print(scores)
###########
'''''
[0.9177888975244394, 0.9234491656935262, 0.9171304174720398, 0.9201481164201938]
Kflod Score:
mean=9196291492775498
0.9201481164201938]

'''''


#KNN GridsearchCV_checkin Parm for got score : 0.10629933402049734
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
model_KNN = KNeighborsRegressor()
grid_params_KNN = {'n_neighbors': [1,2,3,4,5,6,7,8,9,10,11]}

clf_KNN = GridSearchCV(model_KNN, grid_params_KNN, scoring='r2', cv=4)

clf_KNN.fit(X_train,Y_train)
clf_KNN.best_params_ #n_neighbors = 3
clf_KNN.best_estimator_
clf_KNN.best_score_ #CV 0.939252175628826

#import model with best_p of 3 accurcy 93.
model_KNN = KNeighborsRegressor(n_neighbors = 3)
model_KNN.fit(X_train,Y_train)
y_pred = model_KNN.predict(X_test)
#compare both score for checking the how good is the model is -overfitting 
model_KNN.score(X_test,Y_test)
model_KNN.score(X_train,Y_train)
model_KNN.score(Y_test,y_pred)

#how much you effect alpha to minmize the w where u requalizintion 
from sklearn.linear_model import SGDRegressor
model_SGDR = SGDRegressor(loss='squared_loss')
grid_params={'eta0':[0.00001.0001,0.001],
            'learning_rate':['constant','optimal','invscaling'],
                             'alpha':[0.0000001,0.00001,0.00001]}

'''
clf_SGDR.best_score_ #CV means Out[57]: 
SGDRegressor(alpha=0.0001, average=False, early_stopping=False, epsilon=0.1,
             eta0=0.001, fit_intercept=True, l1_ratio=0.15,
             learning_rate='constant', loss='squared_loss', max_iter=1000,
             n_iter_no_change=5, penalty='l2', power_t=0.25, random_state=None,
             shuffle=True, tol=0.001, validation_fraction=0.1, verbose=0,
             warm_start=False)

Out[58]: 0.9183967446859295
'''
clf_SGDR = GridSearchCV(model_SGDR, grid_params, scoring='r2', cv=4)#scoring for classifier is accuracy and r-square for regression
clf_SGDR.fit(X_train,Y_train)
clf_SGDR.best_params_ #{'alpha': 0.001, 'eta0': 0.01, 'learning_rate': 'constant'}
clf_SGDR.best_estimator_
clf_SGDR.best_score_ #CV means 

#apply using the gridsv optimizing parm
clf_SGDR_b = SGDRegressor(loss='squared_loss', learning_rate = 'constant', eta0 = 0.001, alpha=0.0001) 

clf_SGDR_b.fit(X_train,Y_train)
clf_SGDR_b.score(X_test,Y_test)

#Adaboost
'''
weak regressor even with bosted
clf_model.best_params_ 
Out[69]: {'learning_rate': 0.02, 'n_estimators': 110}

clf_model.best_score_

Out[70]: 0.8772176690021309 


'''
from sklearn.ensemble import AdaBoostRegressor
model=AdaBoostRegressor(DecisionTreeRegressor(max_depth=3))
param_dict={
           'n_estimators':range(100,200,10),
           'learning_rate':[0.02,0.03,0.04,0.05,0.06,0.07]
           }
clf_model=GridSearchCV(model,param_dict,cv=4,scoring='r2')
clf_model.fit(X_train,Y_train)
clf_model.best_params_ 
clf_model.best_score_


Aboost_model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=3),learning_rate = 0.02, n_estimators = 110)
Aboost_model.fit(X_train,Y_train)
Aboost_model.score(X_test, Y_test)

########Apply Random Forest######################################
'''
Out[78]: {'max_depth': 9, 'n_estimators': 180}

clf_model.best_score_
Out[79]: 0.9523561513872565
The best Regresser score can be improved according of incresing max_depth and estemerhod
'''
from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor()
params_dict={
           'n_estimators':range(150,250,10),
           'max_depth':[3,5,7,9,]
           }
clf_model=GridSearchCV(model,params_dict,cv=4,scoring='r2')
clf_model.fit(X_train,Y_train)
clf_model.best_params_ # {'max_depth': 9, 'n_estimators 180'}
clf_model.best_score_

model_Rf = RandomForestRegressor( max_depth= 9, n_estimators = 180)
model_Rf.fit(X_train,Y_train)
model_Rf.score(X_test,Y_test)

##################SVM###################
######taking soooo loooong long need more ram more than 3 hours 


from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

param_grid = {'kernel':['rbf', 'poly','linear']}
#gama is what u try to minmize the w by how the distance between points are they '''
'''
GridSearchCV(cv=4, error_score='raise-deprecating',
             estimator=SVR(C=1.0, cache_size=200, coef0=0.0, degree=3,
                           epsilon=0.1, gamma=0.2, kernel='rbf', max_iter=-1,
                           shrinking=True, tol=0.001, verbose=False),
             iid='warn', n_jobs=None,
             param_grid={'kernel': ['rbf', 'poly', 'linear']},
             pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
             scoring=None, verbose=0)

model.best_params_
Out[84]: {'kernel': 'poly'}

clf.score(X_test,Y_test)
Out[88]: 0.9698689252989894

'''
model = GridSearchCV(SVR(gamma=0.2),param_grid, cv=4)
model.fit(X_train,Y_train)
model.best_params_ 

clf = SVR(kernel='poly')
clf.fit(X_train,Y_train)
clf.score(X_test,Y_test)









