#Applying ridge and lasso on processed data

%matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import mean_squared_error




df = pd.read_csv("df_processed.csv")

train = df[df.train_test__train ==1]
y_train=train.SalePrice
X_train = train.drop(['SalePrice','train_test__train'], axis=1)


test = df[df.train_test__train ==0]
X_test = test.drop(['SalePrice','train_test__train'], axis=1)



alphas = 10**np.linspace(10,-2,100)*0.5
alphas.shape


ridge = Ridge(normalize = True)
coefs = []

for a in alphas:
    ridge.set_params(alpha = a)
    ridge.fit(X_train, y_train)
    coefs.append(ridge.coef_)
    
np.shape(coefs)


ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')



ridge2 = Ridge(alpha = 4, normalize = True)
ridge2.fit(X_train, y_train)             # Fit a ridge regression on the training data
pred2 = ridge2.predict(X_test)           # Use this model to predict the test data
print(pd.Series(ridge2.coef_, index = X_train.columns)) # Print coefficients



#neg_mean_squared_error
#rmsle_score


ridgecv = RidgeCV(alphas = alphas, scoring = rmsle_score, normalize = True)
ridgecv.fit(X_train, y_train)
ridgecv.alpha_


ridge4 = Ridge(alpha = ridgecv.alpha_, normalize = True)
ridge4.fit(X_train, y_train)
ridge4.predict(X_test)



submission = pd.read_csv("sample_submission.csv")
submission['SalePrice'] = ridge4.predict(X_test)
submission.to_csv('submission.csv', index=False)
#Score=0.21274



##################################################################################
##################################################################################


lasso = Lasso(max_iter = 10000, normalize = True)
coefs = []

for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(scale(X_train), y_train)
    coefs.append(lasso.coef_)
    
ax = plt.gca()
ax.plot(alphas*2, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')


lassocv = LassoCV(alphas = None, cv = 10, max_iter = 100000, normalize = True)
lassocv.fit(X_train, y_train)

lasso.set_params(alpha=lassocv.alpha_)
lasso.fit(X_train, y_train)


# Some of the coefficients are now reduced to exactly zero.
pd.Series(lasso.coef_, index=X_train.columns)


submission = pd.read_csv("sample_submission.csv")
submission['SalePrice'] = lasso.predict(X_test)
submission.to_csv('submission.csv', index=False)

#Score = 0.154








