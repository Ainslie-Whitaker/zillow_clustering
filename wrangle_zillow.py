#!/usr/bin/env python
# coding: utf-8

# # Zillow

# ### Wrangling the Zillow data
# 
# #### Acquires and prepares Zillow data

# ## Imports

# In[1]:


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

# In[2]:


def get_connection(db, user=env.user, host=env.host, password=env.password):
    '''
    This function takes in user credentials from an env.py file and a database name and creates a connection to the Codeup database through a connection string 
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'


# In[3]:


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


# In[4]:


def query_zillow_data():
    '''
    This function uses the get_connection function to connect to the zillow database and returns the zillow_sql_query read into a pandas dataframe
    '''
    return pd.read_sql(zillow_sql_query,get_connection('zillow'))


# In[5]:



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


# In[8]:


df.head()


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


# In[36]:


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
    df = df[df.taxvaluedollarcnt < 5_000_000]
    df[df.calculatedfinishedsquarefeet < 8000]
    
    
    # drop nulls to make sure none were missed
    df = df.dropna()
    
    # rename columns for clarity
    df.rename(columns={'bathroomcnt':'bathrooms', 'bedroomcnt':'bedrooms', 'calculatedfinishedsquarefeet':'area',
                       'fips':'counties', 'lotsizesquarefeet':'lot_area','propertycountylandusecode':'landusecode',
                       'structuretaxvaluedollarcnt':'structuretaxvalue', 'taxvaluedollarcnt':'taxvalue',
                       'landtaxvaluedollarcnt':'landtaxvalue','propertylandusedesc':'landusedesc'}, inplace=True)
    
    return df
    


# In[39]:


# df = wrangle_zillow()
# df.head()


# In[40]:


# df.columns


# ## Split

# In[12]:


def split_data(df):
    # train/validate/test split
    # splits the data for modeling and exploring, to prevent overfitting
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    
    # Use train only to explore and for fitting
    # Only use validate to validate models after fitting on train
    # Only use test to test best model 
    return train, validate, test 


# In[14]:


# train, validate, test = split_data(df)


# In[16]:


# train.head()


# ## Scale

# In[19]:


def min_max_scaler(train, valid, test):
    '''
    Uses the train & test datasets created by the split_my_data function
    Returns 3 items: mm_scaler, train_scaled_mm, test_scaled_mm
    This is a linear transformation. Values will lie between 0 and 1
    '''
    num_vars = list(train.select_dtypes('number').columns)
    scaler = MinMaxScaler(copy=True, feature_range=(0,1))
    train[num_vars] = scaler.fit_transform(train[num_vars])
    valid[num_vars] = scaler.transform(valid[num_vars])
    test[num_vars] = scaler.transform(test[num_vars])
    return scaler, train, valid, test


# In[34]:


# scaler, train_scaled, validate_scaled, test_scaled = min_max_scaler(train, validate, test)


# In[25]:


# train_scaled.head()


# In[ ]:




