import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.preprocessing
import os
import env

########## Acquire ##########

def get_connection(db, user=env.user, host=env.host, password=env.password):
    '''
    This function takes in user credentials from an env.py file and a database name and creates a connection to the Codeup database through a connection string 
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

zillow_sql_query =  '''
                    select *
                    from properties_2017
                    join predictions_2017 using(parcelid)
                    join propertylandusetype using(propertylandusetypeid)
                    where propertylandusedesc = 'Single Family Residential'
                    and transactiondate like '2017%%';
                    '''

def query_zillow_data():
    '''
    This function uses the get_connection function to connect to the zillow database and returns the zillow_sql_query read into a pandas dataframe
    '''
    return pd.read_sql(zillow_sql_query,get_connection('zillow'))

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

########## Clean ##########

def handle_missing_values(df, col_min_required_pct = .5, row_min_required_pct = .7):
    '''
    This function takes in a dataframe and percentage requirements for both columns and rows and returns a dataframe
    with columns and rows removed that are missing more than those percentages of their values.
    '''
    # specify threshhold for column values
    threshold = int(round(col_min_required_pct*len(df.index),0))
    # drop columns with less than the specified percentage of values
    df.dropna(axis=1, thresh=threshold, inplace=True)
    # specify threshhold for row values
    threshold = int(round(row_min_required_pct*len(df.columns),0))
    # drop rows with less than the specified percentage of values
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df

