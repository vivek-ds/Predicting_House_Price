#Using the procdssed data created n previous code to build machine learning models


#Importing all reqd packages and processed datasets

import os
os.getcwd()        
os.chdir("C:\\Users\\dell\\Desktop\\mmm\\housing_price_kaggle")



import numpy as np # linear algebra
import pandas as pd

pd.set_option('display.width', 1000)

# Plotting Tools
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns

# Import Sci-Kit Learn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, StratifiedKFold, learning_curve, KFold


# pip install xgboost

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Package for stacking models
from vecstack import stacking

import os
os.getcwd()        
os.chdir("C:\\Users\\dell\\Desktop\\mmm\\housing_price_kaggle")
                
train1 = pd.read_csv("train_processed.csv")
test1 = pd.read_csv("test_processed.csv")


###########################################################################
###########################################################################

#Applyng PCA on processed data (without removing high vif vars)


%matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale 
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression, PLSSVD
from sklearn.metrics import mean_squared_error


df = pd.read_csv("df_processed.csv")

train = df[df.train_test__train ==1]
y_train=train.SalePrice
X_train = train.drop(['SalePrice','train_test__train'], axis=1)


test = df[df.train_test__train ==0]
X_test = test.drop(['SalePrice','train_test__train'], axis=1)

#pca = PCA()
#X_reduced = pca.fit_transform(scale(X_train))
#pd.DataFrame(pca.components_.T).loc[:4,:5]
#X_train.shape
#pca.components_.shape

pca2 = PCA()

# Scale the data
X_reduced_train = pca2.fit_transform(scale(X_train))
n = len(X_reduced_train)

# 10-fold CV, with shuffle
kf_10 = model_selection.KFold( n_splits=10, shuffle=True, random_state=1)




#Defining loss function
from sklearn.metrics import make_scorer

def rmsle(real, predicted):
    sum=0.0
    for x in range(len(predicted)):
        if predicted[x]<0 or real[x]<0: #check for negative values
            continue
        p = np.log(predicted[x]+1)
        r = np.log(real[x]+1)
        sum = sum + (p - r)**2
    return (sum/len(predicted))**0.5

rmsle_score = make_scorer(rmsle, greater_is_better=False)





mse = []

regr = LinearRegression()
# Calculate MSE with only the intercept (no principal components in regression)
score = -1*model_selection.cross_val_score(regr, np.ones((n,1)), y_train.ravel(), cv=kf_10, scoring=rmsle_score).mean()    
mse.append(score)

# Calculate MSE using CV for the 19 principle components, adding one component at the time.
for i in np.arange(1, 20):
    score = -1*model_selection.cross_val_score(regr, X_reduced_train[:,:i], y_train.ravel(), cv=kf_10, scoring=rmsle_score).mean()
    mse.append(score)

plt.plot(np.array(mse), '-v')
plt.xlabel('Number of principal components in regression')
plt.ylabel('MSE')
plt.title('SalePrice')
plt.xlim(xmin=-1);




np.cumsum(np.round(pca2.explained_variance_ratio_, decimals=4)*100)
# top 100 PCs describe ~80% of var, but top 10 decribe only ~24% var


X_reduced_test = pca2.transform(scale(X_test))[:,:100]

# Train regression model on training data 
regr = LinearRegression()
regr.fit(X_reduced_train[:,:100], y_train)

# Prediction with test data
pred = regr.predict(X_reduced_test)
#mean_squared_error(y_test, pred)


submission = pd.read_csv("sample_submission.csv")
submission['SalePrice'] = regr.predict(X_reduced_test)
submission.to_csv('submission.csv', index=False)

#Submitting it on kaggle to get the score on test data
#0.1847 with 10 PCs
#0.1847 with 100 PCs




###########################################################################
###########################################################################
# PLS now (again without vif)


n = len(X_train)

# 10-fold CV, with shuffle
kf_10 = model_selection.KFold(n_splits=10, shuffle=True, random_state=1)

mse = []

for i in np.arange(1, 10):
    pls = PLSRegression(n_components=i)
    score = model_selection.cross_val_score(pls, scale(X_train), y_train, cv=kf_10, scoring='neg_mean_squared_error').mean()
    mse.append(-score)

# Plot results
plt.plot(np.arange(1, 10), np.array(mse), '-v')
plt.xlabel('Number of principal components in regression')
plt.ylabel('MSE')
plt.title('Sale Price')
plt.xlim(xmin=-1)


pls = PLSRegression(n_components=6)
pls.fit(scale(X_train), y_train)



submission = pd.read_csv("sample_submission.csv")
submission['SalePrice'] = pls.predict(scale(X_test))
submission.to_csv('submission.csv', index=False)

#score = .17144 without VIF


###########################################################################
###########################################################################

# Training PLS on data after VIF

train = pd.read_csv("train_processed.csv")
test = pd.read_csv("test_processed.csv")

y_train=train.SalePrice
X_train = train.drop('SalePrice', axis=1)
X_test = test


n = len(X_train)

# 10-fold CV, with shuffle
kf_10 = model_selection.KFold(n_splits=10, shuffle=True, random_state=1)

mse = []

for i in np.arange(1, 10):
    pls = PLSRegression(n_components=i)
    score = model_selection.cross_val_score(pls, scale(X_train), y_train, cv=kf_10, scoring='neg_mean_squared_error').mean()
    mse.append(-score)

# Plot results
plt.plot(np.arange(1, 10), np.array(mse), '-v')
plt.xlabel('Number of principal components in regression')
plt.ylabel('MSE')
plt.title('Sale Price')
plt.xlim(xmin=-1)


pls = PLSRegression(n_components=6)
pls.fit(scale(X_train), y_train)



submission = pd.read_csv("sample_submission.csv")
submission['SalePrice'] = pls.predict(scale(X_test))
submission.to_csv('submission.csv', index=False)


#score = .229 after VIF





###########################################################################
###########################################################################
#Applying random forest before PCA


X = train1.drop('SalePrice', axis=1)
y = np.ravel(np.array(train1[['SalePrice']]))
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(np.log(y), np.log(y_pred)))

# Initialize the model
random_forest = RandomForestRegressor(n_estimators=50,
                                      max_depth=5,
                                      min_samples_split=5,
                                      min_samples_leaf=5,
                                      max_features=15,
                                      random_state=42,
                                      oob_score=True
                                     )
# Fit the model to our data
random_forest.fit(X_train, y_train)

# Make predictions on test data
rf_pred = random_forest.predict(X_test)

#cross validation k fold
from sklearn.model_selection import cross_val_score
all_accuracies = cross_val_score(estimator=random_forest, X=X_train, y=y_train, cv=5,scoring = "neg_mean_squared_log_error")
print(all_accuracies.mean())







##########################################################################################
##########################################################################################
#Applying random forest after PCA


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state = 42)
rf.get_params()


X_train = X_reduced_train[:,:100]
X_test = X_reduced_test[:,:100]




from sklearn.model_selection import RandomizedSearchCV
random_grid = {
    'n_estimators': [50, 100, 250],
    'bootstrap': [True, False],
    'max_depth' : [3,5,7],
    'criterion' : ['mae'],
    'min_samples_leaf': [3,5]
}


#neg_mean_squared_log_error
#rmsle_score

rf_random = RandomizedSearchCV(estimator = rf, refit = True, param_distributions = random_grid, n_iter = 2, cv = 5, verbose=2, scoring=rmsle_score, random_state=42, n_jobs = -1)

rf_random.fit(X_train, y_train)

print(rf_random.best_params_)

random_forest = RandomForestRegressor(n_estimators=100,
                                      max_depth=7,
                                      criterion='mae',
                                      min_samples_leaf=3,
                                      max_features=15,
                                      random_state=42,
                                      bootstrap=True
                                     )
# Fit the model to our data
random_forest.fit(X_train, y_train)

# Make predictions on test data
pred = random_forest.predict(X_test)


submission = pd.read_csv("sample_submission.csv")
submission['SalePrice'] = pred
submission.to_csv('submission.csv', index=False)


##########################################################################################
##########################################################################################



#tuning grid with cross validation
from sklearn.model_selection import GridSearchCV

help(RandomForestRegressor)

grid_param = {
    'n_estimators': [50, 100, 250],
    'bootstrap': [True, False],
    'maxdepth' : [3,5,7],
    'criteria' : ['mae'],
    'max_leaf_nodes': [3,5]
}


help(GridSearchCV)
gd_sr = GridSearchCV(estimator=RandomForestRegressor(),
                     param_grid=grid_param,
                     cv=5,
                     n_jobs=-1)

help(gd_sr.fit)
gd_sr.fit(X_train, y_train)

best_parameters = gd_sr.best_params_
print(best_parameters)





# Perform cross-validation to see how well our model does 
kf = KFold(n_splits=5)
y_pred = cross_val_score(random_forest, X, y, cv=kf, n_jobs=-1)
y_pred.mean()



# Initialize our model
xg_boost = XGBRegressor( learning_rate=0.01,
                         n_estimators=6000,
                         max_depth=4, min_child_weight=1,
                         gamma=0.6, subsample=0.7,
                         colsample_bytree=0.2,
                         objective='reg:linear', nthread=-1,
                         scale_pos_weight=1, seed=27,
                         reg_alpha=0.00006
                       )

# Perform cross-validation to see how well our model does 
kf = KFold(n_splits=5)
y_pred = cross_val_score(xg_boost, X, y, cv=kf, n_jobs=-1)
y_pred.mean()


# Fit our model to the training data
xg_boost.fit(X,y)

# Make predictions on the test data
xgb_pred = xg_boost.predict(test)




g_boost = GradientBoostingRegressor( n_estimators=6000, learning_rate=0.01,
                                     max_depth=5, max_features='sqrt',
                                     min_samples_leaf=15, min_samples_split=10,
                                     loss='ls', random_state =42
                                   )

# Perform cross-validation to see how well our model does 
kf = KFold(n_splits=5)
y_pred = cross_val_score(g_boost, X, y, cv=kf, n_jobs=-1)
y_pred.mean()

# Fit our model to the training data
g_boost.fit(X,y)

# Make predictions on test data
gbm_pred = g_boost.predict(test)



# Initialize our model
lightgbm = LGBMRegressor(objective='regression', 
                                       num_leaves=6,
                                       learning_rate=0.01, 
                                       n_estimators=6400,
                                       verbose=-1,
                                       bagging_fraction=0.80,
                                       bagging_freq=4, 
                                       bagging_seed=6,
                                       feature_fraction=0.2,
                                       feature_fraction_seed=7,
                                    )

# Perform cross-validation to see how well our model does
kf = KFold(n_splits=5)
y_pred = cross_val_score(lightgbm, X, y, cv=kf)
print(y_pred.mean())


# Fit our model to the training data
lightgbm.fit(X,y)

# Make predictions on test data
lgb_pred = lightgbm.predict(test)







# List of the models to be stacked
models = [g_boost, xg_boost, lightgbm, random_forest]
# Perform Stacking
S_train, S_test = stacking(models,
                           X_train, y_train, X_test,
                           regression=True,
                           mode='oof_pred_bag',
                           metric=rmse,
                           n_folds=5,
                           random_state=25,
                           verbose=2
                          )


# Initialize 2nd level model
xgb_lev2 = XGBRegressor(learning_rate=0.1, 
                        n_estimators=500,
                        max_depth=3,
                        n_jobs=-1,
                        random_state=17
                       )

# Fit the 2nd level model on the output of level 1
xgb_lev2.fit(S_train, y_train)



# Make predictions on the localized test set
stacked_pred = xgb_lev2.predict(S_test)
print("RMSE of Stacked Model: {}".format(rmse(y_test,stacked_pred)))

g_boost_pred = g_boost.predict(X_test)
print("RMSE of g_boost Model: {}".format(rmse(y_test,g_boost_pred)))

xg_boost_pred = xg_boost.predict(X_test)
print("RMSE of xg_boost Model: {}".format(rmse(y_test,xg_boost_pred)))

lightgbm_pred = lightgbm.predict(X_test)
print("RMSE of lightgbm Model: {}".format(rmse(y_test,lightgbm_pred)))

random_forest_pred = random_forest.predict(X_test)
print("RMSE of random_forest Model: {}".format(rmse(y_test,random_forest_pred)))




y1_pred_L1 = models[0].predict(test)
y2_pred_L1 = models[1].predict(test)
y3_pred_L1 = models[2].predict(test)
y4_pred_L1 = models[3].predict(test)
S_test_L1 = np.c_[y1_pred_L1, y2_pred_L1, y3_pred_L1, y4_pred_L1]


test_stacked_pred = xgb_lev2.predict(S_test_L1)

# Save the predictions in form of a dataframe
submission = pd.DataFrame()

submission['Id'] = np.array(test.index)
submission['SalePrice'] = test_stacked_pred

submission.to_csv('submission.csv', index=False)






g_boost_pred = g_boost.predict(test)

submission = pd.DataFrame()

submission['Id'] = np.array(test.index)
submission['SalePrice'] = g_boost_pred

submission.to_csv('submission.csv', index=False)


