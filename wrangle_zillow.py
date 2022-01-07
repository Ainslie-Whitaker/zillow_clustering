#!/usr/bin/env python
# coding: utf-8

# # Zillow

# ### Wrangling the Zillow data
# 
# #### Acquires and prepares Zillow data

# ## Imports

# In[37]:


import numpy as np
import pandas as pd

# Visualizing
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

# default pandas decimal number display format
pd.options.display.float_format = '{:.2f}'.format

# Split 
from sklearn.model_selection import train_test_split

# Scale
from sklearn.preprocessing import MinMaxScaler
import sklearn.preprocessing

# Stats
import scipy.stats as stats

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

# My Files
import env

# Impoert OS
import os


# ## Acquire

# In[3]:


def get_connection(db, user=env.user, host=env.host, password=env.password):
    '''
    This function takes in user credentials from an env.py file and a database name and creates a connection to the Codeup database through a connection string 
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'


# In[4]:


zillow_sql_query =  '''

    SELECT prop.*, 
           pred.logerror, 
           pred.transactiondate, 
           air.airconditioningdesc, 
           arch.architecturalstyledesc, 
           build.buildingclassdesc, 
           heat.heatingorsystemdesc, 
           landuse.propertylandusedesc, 
           story.storydesc, 
           construct.typeconstructiondesc 
    FROM   properties_2017 prop  
           INNER JOIN (SELECT parcelid,
                       Max(transactiondate) transactiondate 
                       FROM   predictions_2017 
  
                       GROUP  BY parcelid) pred 
                   USING (parcelid)
                   
                            JOIN predictions_2017 as pred USING (parcelid, transactiondate)
           LEFT JOIN airconditioningtype air USING (airconditioningtypeid) 
           LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid) 
           LEFT JOIN buildingclasstype build USING (buildingclasstypeid) 
           LEFT JOIN heatingorsystemtype heat USING (heatingorsystemtypeid) 
           LEFT JOIN propertylandusetype landuse USING (propertylandusetypeid) 
           LEFT JOIN storytype story USING (storytypeid) 
           LEFT JOIN typeconstructiontype construct USING (typeconstructiontypeid) 
    WHERE  prop.latitude IS NOT NULL 
           AND prop.longitude IS NOT NULL
           AND pred.id IN (SELECT MAX(id)
           FROM predictions_2017
           GROUP BY parcelid
           HAVING MAX(transactiondate));
       
       
'''


# In[5]:


def query_zillow_data():
    '''
    This function uses the get_connection function to connect to the zillow database and returns the zillow_sql_query read into a pandas dataframe
    '''
    return pd.read_sql(zillow_sql_query,get_connection('zillow'))


# In[6]:



def get_zillow_data():
    '''
    This function checks for a local zillow.csv file and reads it into a pandas dataframe, if it exists. If not, it uses the get_connection & query_zillow_data functions to query the data and write it locally to a csv file
    '''
    # If csv file exists locally, read in data from csv file.
    if os.path.isfile('zillow.csv'):
        df = pd.read_csv('zillow.csv', index_col=0)
        
    else:
        
        # Query and read data from zillow database
        df = query_zillow_data()
        
        # Cache data
        df.to_csv('zillow.csv')
        
    return df


# In[7]:


df = get_zillow_data()


# In[53]:


# df.head()


# ## Prepare

# In[9]:


# a function to drop missing values based on a threshold
def handle_missing_values(df, prop_required_row = 0.5, prop_required_col = 0.5):
    ''' function which takes in a dataframe, required notnull proportions of non-null rows and columns.
    drop the columns and rows columns based on theshold:'''
    
    #drop columns with nulls
    threshold = int(prop_required_col * len(df.index)) # Require that many non-NA values.
    df.dropna(axis = 1, thresh = threshold, inplace = True)
    
    #drop rows with nulls
    threshold = int(prop_required_row * len(df.columns)) # Require that many non-NA values.
    df.dropna(axis = 0, thresh = threshold, inplace = True)
    
    
    return df


# In[23]:


def wrangle_zillow():
    # read saved .csv
    df = pd.read_csv('zillow.csv')
    
    # propertylandusetypeid that can be considered "single unit" to df
    single_unit = [261, 262, 263, 264, 268, 273, 275, 276, 279]
    df = df[df.propertylandusetypeid.isin(single_unit)]
    
    # df with bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet > 0
    df = df[(df.bedroomcnt > 0) & (df.bathroomcnt > 0) & (df.calculatedfinishedsquarefeet>0)]

    
    # drop missing values based on a threshold
    df = handle_missing_values(df)
   
    # drop unnecessary columns
    df = df.drop(columns=['buildingqualitytypeid', 'parcelid', 'id','calculatedbathnbr', 'finishedsquarefeet12', 'fullbathcnt', 'heatingorsystemtypeid', 'propertyzoningdesc', 'censustractandblock','propertycountylandusecode', 'propertylandusetypeid', 'propertylandusedesc', 'unitcnt','heatingorsystemdesc', 'rawcensustractandblock', 'Unnamed: 0'])
    
    
    # drop null rows for specific columns only
    df = df[df.regionidzip.notnull()]
    df = df[df.yearbuilt.notnull()]
    df = df[df.structuretaxvaluedollarcnt.notnull()]
    df = df[df.taxvaluedollarcnt.notnull()]
    df = df[df.landtaxvaluedollarcnt.notnull()]
    df = df[df.taxamount.notnull()]

    # fill NaNs with mode
    df.lotsizesquarefeet.mode()[0]
    df['lotsizesquarefeet'] = df.lotsizesquarefeet.fillna(df.lotsizesquarefeet.mode()[0])
    df.regionidcity.mode()[0]
    df['regionidcity'] = df.regionidcity.fillna(df.regionidcity.mode()[0])

    
    # replace fips number with city they represent for readability
    df.fips = df.fips.replace({6037:'los_angeles',6059:'orange',6111:'ventura'})
    
    # create dummy variables for fips column and concatenate back onto original dataframe
    dummy_df = pd.get_dummies(df['fips'])
    df = pd.concat([df, dummy_df], axis=1)
    
    # add an age column based on year built
    df['age'] = 2017 - df.yearbuilt.astype(int)

    # bin ages
    df['age_bin'] = pd.cut(df.age, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 130, 140])
    
    # change dataypes where it makes sense
    int_col_list = ['bedroomcnt', 'calculatedfinishedsquarefeet', 'latitude', 'longitude', 'lotsizesquarefeet',  'structuretaxvaluedollarcnt', 'taxvaluedollarcnt', 'landtaxvaluedollarcnt', 'taxamount']
    obj_col_list = ['regionidcounty', 'regionidzip','assessmentyear']

    for col in df:
        if col in int_col_list:
            df[col] = df[col].astype(int)
        if col in obj_col_list:
            df[col] = df[col].astype(int).astype(object)
    
    # ratio of bathrooms to bedrooms
    df['bath_bed_ratio'] = df.bathroomcnt/df.bedroomcnt

    
    # check for outliers
    df = df[df.taxvaluedollarcnt < 2_000_000]
    df[df.calculatedfinishedsquarefeet < 8000]
    
    
    # drop nulls to make sure none were missed
    df = df.dropna()
    
    # rename columns for clarity
    df.rename(columns={'bathroomcnt':'bathrooms', 'bedroomcnt':'bedrooms', 'calculatedfinishedsquarefeet':'area',
                       'fips':'counties', 'lotsizesquarefeet':'lot_area','propertycountylandusecode':'landusecode',
                       'structuretaxvaluedollarcnt':'structuretaxvalue', 'taxvaluedollarcnt':'taxvalue',
                       'landtaxvaluedollarcnt':'landtaxvalue','propertylandusedesc':'landusedesc'}, inplace=True)
    
    return df


# In[24]:


df = wrangle_zillow()
df.head()


# In[52]:


# df.columns


# In[51]:


# df.bedrooms.describe()


# ## Split

# In[27]:


def split_data(df):
    # train/validate/test split
    # splits the data for modeling and exploring, to prevent overfitting
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    
    # Use train only to explore and for fitting
    # Only use validate to validate models after fitting on train
    # Only use test to test best model 
    return train, validate, test 


# In[49]:


# train, validate, test = split_data(df)


# In[50]:


# train.head()


# ## Split into X and y variables

# In[30]:


def split_tvt_into_variables(train, validate, test, target):

# split train into X (dataframe, drop target) & y (series, keep target only)
    X_train = train.drop(columns=[target, 'counties','regionidcounty','regionidzip',
                                    'assessmentyear','transactiondate','age_bin'])
    y_train = train[target]
    
    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=[target, 'counties','regionidcounty','regionidzip',
                                    'assessmentyear','transactiondate','age_bin'])
    y_validate = validate[target]
    
    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=[target, 'counties','regionidcounty','regionidzip',
                                    'assessmentyear','transactiondate','age_bin'])
    y_test = test[target]
    
    return train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test


# In[48]:


# Run split_tvt_into_variables / the target is tax_value
# train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test = split_tvt_into_variables(train, validate, test, target='logerror')


# In[47]:


# X_train.shape


# In[46]:


# X_train.head()


# In[54]:


########## Prep ##########

def prep_zillow_for_model(train, validate, test):
    '''
    This function takes in train, validate, and test dataframes, preps them for scaling, scales them and returns train, validate, and test datasets
    ready for clustering and modeling
    '''

     # drop object type columns to prepare for scaling
    train_model = train.drop(columns = ['counties','regionidcounty','regionidzip',
                                    'assessmentyear','transactiondate','age_bin'])
    validate_model = validate.drop(columns = ['counties','regionidcounty','regionidzip',
                                    'assessmentyear','transactiondate','age_bin'])
    test_model = test.drop(columns = ['counties','regionidcounty','regionidzip',
                                    'assessmentyear','transactiondate','age_bin'])
    
    # use a function to scale data for modeling
    train_scaled, validate_scaled, test_scaled = scale_data_min_max(train_model, validate_model, test_model)
    
    # split scaled data into X_train and y_train
    X_train = train_scaled.drop(columns='logerror')
    y_train = train_scaled.logerror
    X_validate = validate_scaled.drop(columns='logerror')
    y_validate = validate_scaled.logerror
    X_test = test_scaled.drop(columns='logerror')
    y_test = test_scaled.logerror

    return X_train, y_train, X_validate, y_validate, X_test, y_test


# ## Scale

# In[38]:


def Min_Max_Scaler(X_train, X_validate, X_test):
    """
    Takes in X_train, X_validate and X_test dfs with numeric values only
    Returns scaler, X_train_scaled, X_validate_scaled, X_test_scaled dfs 
    """
    #Fit the thing
    scaler = sklearn.preprocessing.MinMaxScaler().fit(X_train)
    
    #transform the thing
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), index = X_train.index, columns = X_train.columns)
    X_validate_scaled = pd.DataFrame(scaler.transform(X_validate), index = X_validate.index, columns = X_validate.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), index = X_test.index, columns = X_test.columns)
    
    return scaler, X_train_scaled, X_validate_scaled, X_test_scaled


# In[44]:


# scaler, X_train_scaled, X_validate_scaled, X_test_scaled = Min_Max_Scaler(X_train, X_validate, X_test)


# In[45]:


# X_train_scaled.head()


# In[42]:


########## Scale ##########

# create function that scales train, validate, and test datasets using min_maxscaler
def scale_data_min_max(train, validate, test):
    '''
    This function takes in train, validate, and test data sets, scales them using sklearn's Min_MaxScaler
    and returns three scaled data sets
    '''
    # Create the scaler
    scaler = sklearn.preprocessing.MinMaxScaler(copy=True, feature_range=(0,1))

    # Fit scaler on train dataset
    scaler.fit(train)

    # Transform and rename columns for all three datasets
    train_scaled = pd.DataFrame(scaler.transform(train), columns = train.columns.tolist())
    validate_scaled = pd.DataFrame(scaler.transform(validate), columns = train.columns.tolist())
    test_scaled = pd.DataFrame(scaler.transform(test), columns = train.columns.tolist())

    return train_scaled, validate_scaled, test_scaled

