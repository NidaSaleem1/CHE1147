#Imports
import os, sys, urllib.request
import numpy as np
import pprint as pp
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, max_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, cross_val_score

from sklearn.dummy import DummyClassifier, DummyRegressor

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LassoCV

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV, Ridge, ElasticNetCV, ElasticNet
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split

import umap
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform, loguniform
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge

from sklearn.decomposition import PCA
from umap import UMAP
import shap

from io import StringIO
from io import BytesIO

#Setting Random Seed
RANDOM_SEED = 1147
np.random.seed(RANDOM_SEED)

#Reading the dataset from a team member's github
# URL_DATASET = 'https://raw.githubusercontent.com/ardin-hsr/CHE1147-Project/main/dataset%20(1).csv'
# DF = pd.read_csv(URL_DATASET)

DF = pd.read_csv(os.path.join("data", "data.csv"))

#Looking at the dataframe's head
pd.set_option('display.max_columns', None)
DF.head()

#Data Pre-processing
#Checking for NaN Values in the dataframe
DF[DF!='-'].info()

#Replacing the excel's NaN to np.nan
DF = DF.replace('-', np.nan)
#Checking columns with >200 NaN values
DF.isna().sum().sort_values(ascending=False)[DF.isna().sum().sort_values(ascending=False)>200]

#Dropping columns with many NaNs
DF = DF.drop(columns=DF.isna().sum().sort_values(ascending=False)[DF.isna().sum().sort_values(ascending=False)>200].index)
#Dropping rows with NaN values and duplicated values
DF = DF.dropna()
DF = DF.drop_duplicates()

#Checking the dataframe's new information
DF.info()

#Changing suitable columns from datatype object (strings) into floats
object_to_float_columns = ['Average Pore Diameters (nm)', 'Flow Rate (mL/min)', 'Adsorption Time (min)', 'Amine Efficiency (mmol/mmol)', 'CO2 Capacity (mmol/g)']
DF[object_to_float_columns] = DF[object_to_float_columns].astype(float)
#Dropping the DOI column because it would not be useful for modelling
#Dropping the amine efficiency column as it is calculated from the target variable, CO2 capacity
#Dropping the relative humidity column because it mostly contains 0 (most likely actually missing values)
DF = DF.drop(columns=['DOI', 'Amine Efficiency (mmol/mmol)', 'Relative Humidity (%)'])

#Recheking the dataframe's information
DF.info()

#Next, we cleanup the columns with Dtype objects to make sure they are clean and consistent
#Starting with the 'Support' Column, checking unique values
DF['Support'].unique()

#There is a problematic unique entry ' MPS' which is the same as 'MPS'
DF['Support'] = DF['Support'].replace(' MPS', 'MPS')

#Next we check the Amine 1 and Additive 1 column
DF['Amine 1 or Additive 1'].unique()

#There are a few problematic unique entries, illustrated in the following list
DF['Amine 1 or Additive 1'][(DF['Amine 1 or Additive 1'].str.contains(' '))].unique()

#Replacing these entries with their proper counterparts
DF['Amine 1 or Additive 1'] = DF['Amine 1 or Additive 1'].str.strip()
#Checking the results
DF['Amine 1 or Additive 1'].unique()

#Next we check the Amine 2 Column
DF['Amine 2 or Additive 2'].unique()

#We will change the '0' entries into 'None' and make them uniform
DF['Amine 2 or Additive 2'] = DF['Amine 2 or Additive 2'].replace('0', 'None')
DF['Amine 2 or Additive 2'] = DF['Amine 2 or Additive 2'].replace('0 ', 'None')
#Checking the results
DF['Amine 2 or Additive 2'].unique()

#Next we check the column Amine 3
DF['Additive 3'].unique()

#We will change the '0' entries into 'None' and make them uniform
DF['Additive 3'] = DF['Additive 3'].replace('0', 'None')
DF['Additive 3'] = DF['Additive 3'].replace('0 ', 'None')
#Checking the results
DF['Additive 3'].unique()

#Next, we deal with the primary, secondary, and tertiary amine ratios
DF['1°, 2°, 3° Amine Ratio']

#To make this ratio more consistent, we will separate them into two columns, primary amine ratio and secondary amine ratio
#Amines with differently typed ratios but actually have the same value (e.g. 4:2:2 vs 2:1:1) will be turned into the same ratio through normalization
#Tertiary amine ratio removed to prevent perfect colinearity between features
DF[['1° Amine', '2° Amine', '3° Amine']] = DF['1°, 2°, 3° Amine Ratio'].str.split(':', expand=True)
DF = DF.drop(columns=['1°, 2°, 3° Amine Ratio'])
DF[['1° Amine', '2° Amine', '3° Amine']] = DF[['1° Amine', '2° Amine', '3° Amine']].astype(float)
DF['Total Amine'] = DF['1° Amine'] + DF['2° Amine'] + DF['3° Amine']
DF['1° Amine'] = DF['1° Amine'] / DF['Total Amine']
DF['2° Amine'] = DF['2° Amine'] / DF['Total Amine']
DF = DF.drop(columns=['Total Amine'])
DF = DF.drop(columns=['3° Amine'])

#Rechecking the DF Information
DF.info()

#First, we check unique counts of additive 2
DF['Amine 2 or Additive 2'].value_counts()

#We could see that there are a few entries where there are very few instances of said entry
#We could also see that most of samples do not have a secondary amine
#Thus, we decided to make this feature binary. 1 if it has a secondary amine and 0 if it does not have secondary amine.
#Splitting the data later will be stratified based on this feature
DF['Amine 2 or Additive 2'] = (DF['Amine 2 or Additive 2'] != 'None').astype(int)
DF['Amine 2 or Additive 2']

#Next, we check if the same problem exists for Additive 3
DF['Additive 3'].value_counts()

#We decide to drop this column because very few samples have a third additive
DF = DF.drop(columns=['Additive 3', 'Additive 3 to Amine 1 Ratio'])
vc = DF['CO2 Concentration (vol%)'].value_counts()
DF = DF[DF['CO2 Concentration (vol%)'].map(vc) >= 40]

#Next, we will encode information about the Support and Amine 1 columns into its properties
#To do this, we have prepared data saved in a different excel file, we will combine that data into this dataframe
# URL_SUPPORT = "https://raw.githubusercontent.com/ardin-hsr/CHE1147-Project/main/CHE1147_Supports.xlsx"

# RESP_SUPPORT = requests.get(URL_SUPPORT)

DF_SUPPORT = pd.read_excel(os.path.join("data", "CHE1147_Supports.xlsx"))
DF_SUPPORT.head()

#Most of the support's data is already described by existing columns, we just need to add the support's morphology
DF = pd.merge(DF, DF_SUPPORT, on='Support', how='left')
DF = DF.dropna()
DF = DF.drop(columns=['Support'])
DF.head()

#Next, we do the same thing for the amine 1 column
# URL_AMINE = "https://raw.githubusercontent.com/ardin-hsr/CHE1147-Project/main/CHE1147_Amine.xlsx"
# RESP_AMINE = requests.get(URL_AMINE)
# DF_AMINE = pd.read_excel(BytesIO(RESP_AMINE.content))

DF_AMINE = pd.read_excel(os.path.join("data", "CHE1147_Amine.xlsx"))
DF_AMINE = DF_AMINE.rename(columns={'Amine': 'Amine 1 or Additive 1', 'Structure': 'Amine_structure', 'Density (g/cm³)': 'Amine_Density (g/cm³)', 'Viscosity (mPa·s)': 'Amine_Viscosity (mPa·s)'})
DF_AMINE.head()

#Merging the dataframe
DF = pd.merge(DF, DF_AMINE, on='Amine 1 or Additive 1', how='left')
DF = DF.dropna()
DF = DF.drop(columns=['Amine 1 or Additive 1'])
DF = DF.drop_duplicates().dropna()

#Lastly, we will one-hot encode the support and amine morphology/structure columns so that we could use them for machine learning
#We start with the support morphology, we will remove samples whose morphology have very few entries
DF['Support_Morphology'].value_counts()
#We will also use this column for stratification

#Remove entries with less than 20 instances
DF = DF[DF['Support_Morphology'].map(DF['Support_Morphology'].value_counts()) >= 20]

#Next, we do the same for Amine 1 Structure if necessary
DF['Amine_structure'].value_counts()

#Checking final dataframe info
DF.info()

#We now split the data into train and test
from sklearn.model_selection import train_test_split

strata = DF[['CO2 Concentration (vol%)']].astype(str).agg('_'.join, axis=1)

DF_TRAIN, DF_TEST = train_test_split(
    DF,
    test_size=0.1,
    random_state=RANDOM_SEED
)

#Next, we drop the Amine 2 or Additive 2 column (because information about this is encoded in the Amine 2 or Additive 2 to Amine 1 Ratio column)
DF_TRAIN = DF_TRAIN.drop(columns=['Amine 2 or Additive 2'])
#Next, we one hot encode the morphology and structure column for both datasets
DF_TRAIN = pd.get_dummies(DF_TRAIN, columns=['Support_Morphology', 'Amine_structure'])
DF_TRAIN = DF_TRAIN.drop(columns = ['Support_Morphology_Hexagonal', 'Amine_structure_Branched'])
DF_TEST = pd.get_dummies(DF_TEST, columns=['Support_Morphology', 'Amine_structure'])
DF_TEST = DF_TEST.drop(columns = ['Support_Morphology_Hexagonal', 'Amine_structure_Branched'])

#Change the boolean columns into int datatype
DF_TRAIN[DF_TRAIN.select_dtypes(include='bool').columns] = DF_TRAIN[DF_TRAIN.select_dtypes(include='bool').columns].astype(int)
DF_TEST[DF_TEST.select_dtypes(include='bool').columns] = DF_TEST[DF_TEST.select_dtypes(include='bool').columns].astype(int)
#Checking final DF Information
DF_TRAIN.info()

#Checking the final information for the test df
DF_TEST.info()

#Checking that there are no columns with 0 variance
DF_TRAIN.var()[DF_TRAIN.var()==0] + DF_TEST.var()[DF_TEST.var()==0]

#Dropping rows where Target value is missing
DF_TRAIN = DF_TRAIN[DF_TRAIN['CO2 Capacity (mmol/g)'] != 0]

#Store processed data
trainDFLoc = os.path.join("data", "df_train.csv")
testDFLoc = os.path.join("data", "df_test.csv")
cleanedDFLoc = os.path.join("data", "cleaned_df.csv")

DF_TRAIN.to_csv(trainDFLoc, index=False)
DF_TEST.to_csv(testDFLoc, index = False)
DF.to_csv(cleanedDFLoc, index=False)