#!/usr/bin/env python
# coding: utf-8

## Imports
import numpy as np
import pandas as pd

# Visualizing
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

# # default pandas decimal number display format
# pd.options.display.float_format = '{:.2f}'.format

# Split 
from sklearn.model_selection import train_test_split

# Scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import sklearn.preprocessing

# Model
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import TweedieRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoLars
from sklearn.preprocessing import PolynomialFeatures

# Stats
import scipy.stats as stats

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

# My Files
import env

# Impoert OS
import os


########## Acquire ##########


def get_connection(db, user=env.user, host=env.host, password=env.password):
    '''
    This function takes in user credentials from an env.py file and a database name and creates a connection to the Codeup database through a connection string 
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

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



def wrangle_zillow():
    '''
    This function pulls the zillow data in, cleans and preps it, and returns a dataframe
    '''
    # use function to connect and pull in data from the sql server
    df = get_zillow_data()
    
    # propertylandusetypeid that can be considered "single unit" to df
    single_unit = [261, 262, 263, 264, 268, 273, 275, 276, 279]
    df = df[df.propertylandusetypeid.isin(single_unit)]
    
    # df with bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet > 0
    df = df[(df.bedroomcnt > 0) & (df.bathroomcnt > 0) & (df.calculatedfinishedsquarefeet>0)]

    # drop missing values based on a threshold
    df = handle_missing_values(df)
   
    # drop unnecessary columns
    df = df.drop(columns=['buildingqualitytypeid', 'parcelid', 'id','calculatedbathnbr', 'finishedsquarefeet12', 'fullbathcnt', 'heatingorsystemtypeid', 'propertyzoningdesc', 'censustractandblock','propertycountylandusecode', 'propertylandusetypeid', 'propertylandusedesc', 'unitcnt','heatingorsystemdesc', 'rawcensustractandblock'])
    
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

########## Split ##########

def split_data(df):
    # train/validate/test split
    # splits the data for modeling and exploring, to prevent overfitting
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)

    train = train.reset_index(drop=True)
    validate = validate.reset_index(drop=True)
    test = test.reset_index(drop=True)
    
    # Use train only to explore and for fitting
    # Only use validate to validate models after fitting on train
    # Only use test to test best model 
    return train, validate, test



def split_tvt_into_variables(train, validate, test, target):

#    split train into X (dataframe, drop target) & y (series, keep target only)
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

########## Scale ##########

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
    train_scaled = pd.DataFrame(scaler.transform(train), index = train.index, columns = train.columns.tolist())
    validate_scaled = pd.DataFrame(scaler.transform(validate), index = validate.index, columns = train.columns.tolist())
    test_scaled = pd.DataFrame(scaler.transform(test), index = test.index, columns = train.columns.tolist())

    return train_scaled, validate_scaled, test_scaled

########## Modeling ##########

def prep_zillow_for_model(train, validate, test):
    '''
    This function takes in train, validate, and test dataframes, preps them for scaling, scales them and returns train, validate, and test datasets
    ready for clustering and modeling
    '''

     # drop object type columns to prepare for scaling
    train_model2 = train.drop(columns = ['counties','regionidcounty','regionidzip',
                                    'assessmentyear','transactiondate','age_bin'])
    validate_model2 = validate.drop(columns = ['counties','regionidcounty','regionidzip',
                                    'assessmentyear','transactiondate','age_bin'])
    test_model2 = test.drop(columns = ['counties','regionidcounty','regionidzip',
                                    'assessmentyear','transactiondate','age_bin'])
    
    # use a function to scale data for modeling
    train_scaled, validate_scaled, test_scaled = scale_data_min_max(train_model2, validate_model2, test_model2)
    
    # split scaled data into X_train and y_train
    X_train = train_scaled.drop(columns='logerror')
    y_train = pd.DataFrame(train.logerror)
    X_validate = validate_scaled.drop(columns='logerror')
    y_validate = pd.DataFrame(validate.logerror)
    X_test = test_scaled.drop(columns='logerror')
    y_test = pd.DataFrame(test.logerror)

    return X_train, y_train, X_validate, y_validate, X_test, y_test


def prep_zillow():

    df = wrangle_zillow()
    df = df.reset_index(drop=True)
    train, validate, test = split_data(df)
    X_train, y_train, X_validate, y_validate, X_test, y_test = prep_zillow_for_model(train, validate, test)
    return train, X_train, y_train, X_validate, y_validate, X_test, y_test



def create_agetax_cluster(X_train, X_validate, X_test):
    '''
    This function takes in X_train, X_validate, and X_test datasets and creates clusters based on some of the features
    '''
    # select the features to use
    X = X_train[['age', 'taxvalue']]
    X2 = X_validate[['age', 'taxvalue']]
    X3 = X_test[['age', 'taxvalue']]

    #  use KMeans to create 3 clusters to see if that may be more meaningful
    # define the thing
    kmeans = KMeans(n_clusters=3, random_state = 369)

    # fit the thing
    kmeans.fit(X)

    # Use the thing to predict
    kmeans.predict(X)

    # create a new column with the predicted cluster in the original    X_dataframes
    X_train['agetax_cluster'] = kmeans.predict(X)
    X_validate['agetax_cluster'] = kmeans.predict(X2)
    X_test['agetax_cluster'] = kmeans.predict(X3)

    # create dataframe of cluster centers
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns)

    return X_train, X_validate, X_test



def create_bedbath_area_cluster(X_train, X_validate, X_test):
    '''
    This function takes in X_train, X_validate, and X_test datasets and creates clusters based on some of the features
    '''
    # select the features to use
    X = X_train[['bathrooms', 'bedrooms', 'area']]
    X2 = X_validate[['bathrooms', 'bedrooms', 'area']]
    X3 = X_test[['bathrooms', 'bedrooms', 'area']]

    # use KMeans to create 4 clusters
    # define the thing
    kmeans = KMeans(n_clusters=4, random_state = 369)

    # fit the thing
    kmeans.fit(X)

    # Use the thing to predict
    kmeans.predict(X)

    # create a new column with the predicted cluster in the original X_train
    X_train['bedbath_area_cluster'] = kmeans.predict(X)
    X_validate['bedbath_area_cluster'] = kmeans.predict(X2)
    X_test['bedbath_area_cluster'] = kmeans.predict(X3)

    return X_train, X_validate, X_test



def prepare_clusters_for_modeling(X_train, X_validate, X_test):
    '''
    This function takes in an X_train, X_validate, and X_test dataset and preps them and encodes them for modeling
    '''
    X_train, X_validate, X_test = create_agetax_cluster(X_train, X_validate, X_test)
    X_train, X_validate, X_test = create_bedbath_area_cluster(X_train, X_validate, X_test)
    # give clusters names
    X_train.agetax_cluster = X_train.agetax_cluster.map({0: "older_lowtaxvalue", 1: "newer_lowtaxvalue", 2: "all_ages_hightaxvalue"})

    X_validate.agetax_cluster = X_validate.agetax_cluster.map({0: "older_lowtaxvalue", 1: "newer_lowtaxvalue", 2: "all_ages_hightaxvalue"})

    X_test.agetax_cluster = X_test.agetax_cluster.map({0: "older_lowtaxvalue", 1: "newer_lowtaxvalue", 2: "all_ages_hightaxvalue"})

    X_train.bedbath_area_cluster = X_train.bedbath_area_cluster.map({0: "large_3plusbed", 1: "small_2bed", 2: "tiny_1bed", 3: "medium_3bed"})

    X_validate.bedbath_area_cluster = X_validate.bedbath_area_cluster.map({0: "large_3plusbed", 1: "small_2bed", 2: "tiny_1bed", 3: "medium_3bed"})

    X_test.bedbath_area_cluster = X_test.bedbath_area_cluster.map({0: "large_3plusbed", 1: "small_2bed", 2: "tiny_1bed", 3: "medium_3bed"})
    
    # encode cluster columns
    X_train_model = pd.get_dummies(X_train[['agetax_cluster','bedbath_area_cluster']])
    X_validate_model = pd.get_dummies(X_validate[['agetax_cluster','bedbath_area_cluster']])
    X_test_model = pd.get_dummies(X_test[['agetax_cluster','bedbath_area_cluster']])

    return X_train_model, X_validate_model, X_test_model


def create_evaluate_baseline(y_train,y_validate):
    # create a baseline of median logerror
    y_train['baseline'] = y_train.logerror.median()
    y_validate['baseline'] = y_train.logerror.median()

    # RMSE of logerror median
    rmse_train_baseline = round(mean_squared_error(y_train.logerror, y_train.baseline)**(1/2), 5)
    rmse_validate_baseline = round(mean_squared_error(y_validate.logerror, y_validate.baseline)**(1/2), 5)

    return rmse_train_baseline, rmse_validate_baseline

    # print(f'Baseline logerror is {round(y_train.logerror.median(),5)}.')
    # print("RMSE for baseline using Median\nTrain/In-Sample: ", round(rmse_train_baseline, 5), 
    #   "\nValidate/Out-of-Sample: ", round(rmse_validate_baseline, 5))


def create_evaluate_tweedie_regressor(X_train_model, X_validate_model, y_train, y_validate):
    # create the model object
    glm = TweedieRegressor(power=0, alpha=0.01)

    # fit the model to our training data and specify y column 
    glm.fit(X_train_model, y_train.logerror)

    # predict train & validate
    y_train['logerror_pred_glm'] = glm.predict(X_train_model)
    y_validate['logerror_pred_glm'] = glm.predict(X_validate_model) 

    # evaluate train & validate: rmse
    rmse_train_glm = round(mean_squared_error(y_train.logerror, y_train.logerror_pred_glm)**(1/2), 5)
    rmse_validate_glm = round(mean_squared_error(y_validate.logerror, y_validate.logerror_pred_glm)**(1/2), 5)

    return rmse_train_glm, rmse_validate_glm
    # print("RMSE for GLM using Tweedie\nTraining/In-Sample: ", round(rmse_train_glm, 5), 
    #   "\nValidation/Out-of-Sample: ", round(rmse_validate_glm, 5))




def create_evaluate_ordinaryleastsquares(X_train_model, X_validate_model, y_train, y_validate):
    # create the model object
    lm = LinearRegression(normalize=True)

    # fit the model to our training data and specify y column 
    lm.fit(X_train_model, y_train.logerror)

    # predict train & validate
    y_train['logerror_pred_lm'] = lm.predict(X_train_model)
    y_validate['logerror_pred_lm'] = lm.predict(X_validate_model) 

    # evaluate train & validate: rmse
    rmse_train_OLS = round(mean_squared_error(y_train.logerror, y_train.logerror_pred_lm)**(1/2), 5)
    rmse_validate_OLS = round(mean_squared_error(y_validate.logerror, y_validate.logerror_pred_lm)**(1/2), 5)

    return rmse_train_OLS, rmse_validate_OLS
    # print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", round(rmse_train_OLS, 5), 
    #   "\nValidation/Out-of-Sample: ", round(rmse_validate_OLS, 5))



def create_evaluate_subset_ols(X_train_model, X_validate_model, y_train, y_validate):
    # select only bed, bath, area cluster features
    X_train_bba = X_train_model.iloc[:,3:]
    X_validate_bba = X_validate_model.iloc[:,3:]

    # create the model object
    lm3 = LinearRegression(normalize=True)

    # fit the model to our training data and specify y column 
    lm3.fit(X_train_bba, y_train.logerror)

    # predict train & validate
    y_train['logerror_pred_lm3'] = lm3.predict(X_train_bba)
    y_validate['logerror_pred_lm3'] = lm3.predict(X_validate_bba)

    # evaluate train & validate: rmse
    rmse_train_sub1 = round(mean_squared_error(y_train.logerror, y_train.logerror_pred_lm3)**(1/2), 5)  
    rmse_validate_sub1 = round(mean_squared_error(y_validate.logerror, y_validate.logerror_pred_lm3)**(1/2), 5)

    return rmse_train_sub1, rmse_validate_sub1
    # print("RMSE for subset OLS using LinearRegression\nTraining/In-Sample: ", round(rmse_train_sub1, 5), 
    #   "\nValidation/Out-of-Sample: ", round(rmse_validate_sub1, 5))


def create_evaluate_polynomial_regression(X_train_model, X_validate_model, y_train, y_validate):
    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=2)

    # fit X_train_model and transform all dataframes
    X_train_degree2 = pf.fit_transform(X_train_model)
    X_validate_degree2 = pf.transform(X_validate_model)
    # X_test_degree2 = pf.transform(X_test_model)

    # create the model object
    lm2 = LinearRegression(normalize=True)

    # fit the model to our training data and specify y column 
    lm2.fit(X_train_degree2, y_train.logerror)

    # predict train & validate
    y_train['logerror_pred_lm2'] = lm2.predict(X_train_degree2) 
    y_validate['logerror_pred_lm2'] = lm2.predict(X_validate_degree2)

    # evaluate: rmse
    rmse_train_pr = round(mean_squared_error(y_train.logerror, y_train.logerror_pred_lm2)**(1/2), 5)
    rmse_validate_pr = round(mean_squared_error(y_validate.logerror, y_validate.logerror_pred_lm2)**(1/2), 5)

    return rmse_train_pr, rmse_validate_pr
    # print("RMSE for Polynomial Model\nTraining/In-Sample: ", round(rmse_train_pr, 5), 
    #   "\nValidation/Out-of-Sample: ", round(rmse_validate_pr, 5))


def compare_models(X_train_model, X_validate_model, y_train, y_validate):
    rmse_train_baseline, rmse_validate_baseline = create_evaluate_baseline(y_train,y_validate)
    rmse_train_glm, rmse_validate_glm = create_evaluate_tweedie_regressor(X_train_model, X_validate_model, y_train, y_validate)
    rmse_train_OLS, rmse_validate_OLS = create_evaluate_ordinaryleastsquares(X_train_model, X_validate_model, y_train, y_validate)
    rmse_train_sub1, rmse_validate_sub1 = create_evaluate_subset_ols(X_train_model, X_validate_model, y_train, y_validate)
    rmse_train_pr, rmse_validate_pr = create_evaluate_polynomial_regression(X_train_model, X_validate_model, y_train, y_validate)

    data = pd.DataFrame({'RMSE_For': ['Baseline_using_Median', 'GLM_using_Tweedie', 'OLS_using_LinearRegression', 'OLS_using_LinearRegression_subset', 'Polynomial_Model'], 
    'Training/In-Sample': [rmse_train_baseline, rmse_train_glm, rmse_train_OLS, rmse_train_sub1, rmse_train_pr], 
    'Diff_from_Baseline': [rmse_train_baseline-rmse_train_baseline, rmse_train_glm-rmse_train_baseline, 
                            rmse_train_OLS-rmse_train_baseline, rmse_train_sub1-rmse_train_baseline, 
                            rmse_train_pr-rmse_train_baseline],
    'Validation/Out-of-Sample': [rmse_validate_baseline, rmse_validate_glm, rmse_validate_OLS, rmse_validate_sub1, rmse_validate_pr], 
    'Diff_from_Baseline2': [rmse_validate_baseline-rmse_validate_baseline, rmse_validate_glm-rmse_validate_baseline, 
                            rmse_validate_OLS-rmse_validate_baseline, rmse_validate_sub1-rmse_validate_baseline, 
                            rmse_validate_pr-rmse_validate_baseline]})

    return data

def model_test(X_train_model, y_train, y_test, X_test_model):
    # create the model object
    glm = TweedieRegressor(power=0, alpha=0.01)

    # fit the model to our training data and specify y column 
    glm.fit(X_train_model, y_train.logerror)

    # predict tesr
    y_test['logerror_pred_glm'] = glm.predict(X_test_model)

    # evaluate test: rmse
    rmse_test_glm = round(mean_squared_error(y_test.logerror, y_test.logerror_pred_glm)**(1/2), 5)

    return rmse_test_glm

    # Clustering

    # functions to create clusters and scatter-plot:


def create_cluster(train, X, k):
    
    """ Takes in df, X (dataframe with variables you want to cluster on) and k
    # It scales the X, calcuates the clusters and return train (with clusters), the Scaled dataframe,
    #the scaler and kmeans object and unscaled centroids as a dataframe"""
    
    scaler = StandardScaler(copy=True).fit(X)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns.values).set_index([X.index.values])
    kmeans = KMeans(n_clusters = k, random_state = 42)
    kmeans.fit(X_scaled)
    kmeans.predict(X_scaled)
    train['cluster'] = kmeans.predict(X_scaled)
   
#     validate['cluster'] = kmeans.predict(X_scaled)
   
#     test['cluster'] = kmeans.predict(X_scaled)
   
    centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=X_scaled.columns)
    return train, X_scaled, scaler, kmeans, centroids


def create_scatter_plot(x,y,df,kmeans, X_scaled, scaler):
    
    """ Takes in x and y (variable names as strings, along with returned objects from previous
    function create_cluster and creates a plot"""
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x = x, y = y, data = df, hue = 'cluster')
    centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=X_scaled.columns)
    centroids.plot.scatter(y=y, x= x, ax=plt.gca(), alpha=.30, s=500, c='black')