# Preprocessing & checking for multicollinearity using vif


#Loading dataset and reqd packages
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


os.getcwd()        
os.chdir("C:\\Users\\dell\\Desktop\\mmm\\housing_price_kaggle")
                
train = pd.read_csv("train.csv", index_col='Id')
test = pd.read_csv("test.csv", index_col='Id')


####################################################################
####################################################################
#Merging train/test together to do pre-processing together
#Adding Additional column train_test which has train/test identifier

train.shape
test.shape
train['train_test'] = ['train']*train.shape[0]

test['SalePrice'] = [1]*test.shape[0]
test['train_test'] = ['test']*test.shape[0]
  

data_f = pd.concat([train,test])
data_f.shape


####################################################################
####################################################################



# Plot missing values 
def plot_missing(df):
    # Find columns having missing values and count
    missing = df.isnull().sum()
    print(missing)
    
    missing = missing[missing > 0]
    print(missing)
    #missing.sort_values(inplace=True)
    
    # Plot missing values by count 
    missing.plot.bar(figsize=(12,8))
    plt.xlabel('Columns with missing values')
    plt.ylabel('Count')
    
    
plot_missing(train)
plot_missing(test)
plot_missing(data_f)

#Looking at structure of data_f
data_f.head()
data_f.info()
data_f.describe()

len(data_f.columns)
data_f.shape

#Checking the missing values
ar1 = np.array(data_f.columns)
missing = pd.DataFrame(list(data_f.isnull().sum()), index = ar1, columns =['number_miss'])
missing = missing.sort_values(by = 'number_miss' , ascending = False)
print(missing)

#Checking the columns with most missing values
missing.shape
high_missing_cols = data_f.loc[:,list(missing.index[0:5])]
high_missing_cols.shape
high_missing_cols.describe()
high_missing_cols.iloc[:,4].value_counts()

#Removing columnswith high missing rate
varnames = list(missing.index[5:])
data_f = data_f.loc[:,varnames]
data_f.shape



# # IMPUTING MISSING VALUES for rest of the columns
def fill_missing_values(df):
    ''' This function imputes missing values with median for numeric columns 
        and most frequent value for categorical columns'''
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    for column in list(missing.index):
        if df[column].dtype == 'object':
            df[column].fillna(df[column].value_counts().index[0], inplace=True)
        elif df[column].dtype == 'int64' or 'float64' or 'int16' or 'float16':
            df[column].fillna(df[column].median(), inplace=True)


fill_missing_values(data_f)
data_f.isnull().sum().max()



###################################################################
###################################################################

#Creating dummies for categorical columns

print("Train Dtype counts: \n{}".format(data_f.dtypes.value_counts()))

#Below code will help to cout the total dummy vars which will be created and total numerical vars
count=0
count1=0
cat_columns=[]
for col in data_f.columns:
    if data_f[col].dtype == 'object':
        print(col, "\n", data_f[col].value_counts())
        count=count + len((data_f[col].unique()).tolist()) -1
        print(count)
        cat_columns.append(col)
    else:
        print(col, "\n")
        count1 += 1
 
print(cat_columns)    

print(count)
print(count1)

#Count of total dummies to be created & total numerical columns
197+37

#Maintaining a copy of data_f, before creting dummies
data_f1=data_f.copy()




# Categorical boolean mask
categorical_feature_mask = data_f.dtypes==object

# filter categorical columns using mask and turn it into a list
categorical_cols = data_f.columns[categorical_feature_mask].tolist()

cat_columns = categorical_cols
df_processed = pd.get_dummies(data_f, prefix_sep="__",drop_first=True,
                              columns=cat_columns)

#Removing . or any numeric from the colnames of dummies created
import string
df_processed.columns = [''.join(c for c in s if c not in ['.',' ','1','2','3','4','5','6','7','8','9','0']) for s in df_processed.columns]


#Exporting the processed data
df_processed.to_csv('df_processed.csv', index=False)

###################################################################
###################################################################


#Checking for multicollinearity using vif


# Import functions
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Get variables for which to compute VIF
len(list(df_processed.columns))
cols = list(df_processed.columns)
cols.remove('SalePrice')
cols.remove('train_test__train')
X=df_processed.loc[:,cols]

# Compute and view VIF
vif = pd.DataFrame()
vif["variables"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# View results using print
vif = vif.sort_values(by = 'VIF' , ascending = False)
print(vif)
vif.shape
#Top vars are having extremely high vif

#checking vif's again after removing top 4 vars
vif.index = np.arange( 0 , vif.shape[0], step=1)
X = X.loc[:,list(vif.loc[4:,"variables"])]

vif = pd.DataFrame()
vif["variables"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# View results using print
vif = vif.sort_values(by = 'VIF' , ascending = False)
print(vif)
#Top vars are still having extremely high vif



#checking vif's again after removing top 4 vars
vif.index = np.arange( 0 , vif.shape[0], step=1)
X = X.loc[:,list(vif.loc[8:,"variables"])]

vif = pd.DataFrame()
vif["variables"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# View results using print
vif = vif.sort_values(by = 'VIF' , ascending = False)
print(vif)
#Top vars are still having extremely high vif



#Let's write a loop now to check vif's again & again after removing top 4 vars till all vif's are below 10

max_vif_val = 10000

while(max_vif_val > 10):
    vif.index = np.arange( 0 , vif.shape[0], step=1)
    X = X.loc[:,list(vif.loc[5:,"variables"])]

    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    # View results using print
    vif = vif.sort_values(by = 'VIF' , ascending = False)
    max_vif_val = vif.iloc[0,1]
    #print(vif)


print(vif)
#Now all vif's are below 10



#Exporting train/test separately after removing any punctuation

import string
X.columns = [''.join(c for c in s if c not in string.punctuation) for s in X.columns]



X['train_test__train'] = df_processed[['train_test__train']]
X['SalePrice'] = df_processed[['SalePrice']]


cols = list(np.arange(0 , X.shape[1] -2, step=1))
cols.extend([X.shape[1]-1])
train = X.loc[X['train_test__train'] ==1 , :]
train = train.iloc[:,cols]


test = X.loc[X['train_test__train'] ==0 ,:]
test = test.iloc[:,np.arange(0 , X.shape[1] -2, step=1)]


train.to_csv('train_processed.csv', index=False)
test.to_csv('test_processed.csv', index=False)



###################################################################
###################################################################

