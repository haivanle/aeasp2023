# Import libraries
import pandas as pd
import numpy as np
import cpi
import datetime as dt
import matplotlib.pyplot as plt

# Libraries for regression
import statsmodels.formula.api as sm
import statsmodels.api as sm1
from statsmodels.sandbox.regression.gmm import IV2SLS
from statsmodels.api import add_constant
from linearmodels.iv import IV2SLS
from linearmodels.iv import compare

# Read files
execu500 = pd.read_csv('/Users/ASUS/Downloads/exesp500fullvariables.csv')
execu400 = pd.read_csv('/Users/ASUS/Downloads/exesp400fullvariables.csv')
execu600 = pd.read_csv('/Users/ASUS/Downloads/exesp600fullvariables.csv')
sp500 = pd.read_csv('/Users/ASUS/Downloads/sp500afull.csv')
sp400 = pd.read_csv('/Users/ASUS/Downloads/sp400afull.csv')
sp600 = pd.read_csv('/Users/ASUS/Downloads/sp600afull.csv')

# Rename key columns
sp500 = sp500.rename(columns={"fyear": "YEAR", "tic": "TICKER"})
sp400 = sp500.rename(columns={"fyear": "YEAR", "tic": "TICKER"})
sp600 = sp500.rename(columns={"fyear": "YEAR", "tic": "TICKER"})

# Combine executive data
df = pd.concat([execu500, execu400, execu600])
df['stockoptions'] = df['STOCK_AWARDS'] + df['OPTION_AWARDS']

# Adjust Stock Options based on CPI, base year is 2022
df["adjusted_stockoptions"] = df.apply(lambda x: cpi.inflate(x["stockoptions"], x["YEAR"]), axis=1)

# Adjust total compensation based on CPI, base year is 2022
df["adjusted_TC"] = df.apply(lambda x: cpi.inflate(x["TOTAL_SEC"], x["YEAR"]), axis=1)

# Change to million USD
df['adjusted_stockoptions'] = df['adjusted_stockoptions'] / 1000
df['adjusted_TC'] = df['adjusted_TC'] / 1000

# Drop rows with negative compensation
df = df[(df['adjusted_stockoptions']>0) & (df['adjusted_TC']>0)]

# Keep only CEOs data
ceo = df[(df['CEOANN'] == 'CEO')]

# Get the year that executive became CEO
ceo['dateceo'] = pd.to_datetime(ceo['BECAMECEO'])
ceo['yearceo'] = ceo['dateceo'].dt.year

# Get the tenure of CEOs - defined as : year of data observation - yearbecame ceo
ceo['tenure'] = (ceo['YEAR'].astype(float) - ceo['yearceo'])

# Drop negative tenures
ceo_tenure = ceo[(ceo['tenure'] > 0)]

''' Detect outliers in compensation '''

''' Detection '''
# IQR
# Calculate the upper and lower limits
Q1a = ceo_tenure['adjusted_TC'].quantile(0.25)
Q3a = ceo_tenure['adjusted_TC'].quantile(0.75)
IQRa = Q3a - Q1a
lowera = Q1a - 1.5*IQRa
uppera = Q3a + 1.5*IQRa

# IQR
# Calculate the upper and lower limits
Q1b = ceo_tenure['adjusted_stockoptions'].quantile(0.25)
Q3b = ceo_tenure['adjusted_stockoptions'].quantile(0.75)
IQRb = Q3b - Q1b
lowerb = Q1b - 1.5*IQRb
upperb = Q3b + 1.5*IQRb
 
# Removing the outliers
ceo_tenure = ceo_tenure.loc[(ceo_tenure['adjusted_stockoptions']< upperb) & (ceo_tenure['adjusted_stockoptions'] > lowerb) 
                            & (ceo_tenure['adjusted_TC'] < uppera) & (ceo_tenure['adjusted_TC'] > lowera) ]

''' Detect outliers in tenure '''
# IQR
# Calculate the upper and lower limits
Q1_t = ceo_tenure['tenure'].quantile(0.25)
Q3_t = ceo_tenure['tenure'].quantile(0.75)
IQR_t = Q3_t - Q1_t
lower_t = Q1_t - 1.5*IQR_t
upper_t = Q3_t + 1.5*IQR_t
 
# Removing the outliers
ceo_tenure = ceo_tenure.loc[(ceo_tenure['tenure']< upper_t) & (ceo_tenure['tenure'] > lower_t) ]

# Combine financial data
df2 = pd.concat([sp500, sp400, sp600])

# Component of Altman's Z-score for public companies
df2['X1'] = (df2['act'] - df2['lct']) / df2['at']
df2['X2'] = df2['re'] / df2['at']
df2['X3'] = df2['ebit'] / df2['at']
df2['X4'] = df2['prcc_f'] * df2['csho'] / df2['lt']
df2['X5'] = df2['sale'] / df2['at']

# Calculate Z-score
df2['zscore'] = 1.2*df2['X1'] + 1.4*df2['X2'] + 3.3*df2['X3'] + 0.6*df2['X4'] + 1.0*df2['X5']

# Merge executive data with firm's financial data
secondstage = ceo_tenure.merge(df2, on = ['YEAR', 'TICKER'], how = 'left')
secondstage.head()

# Read Chicago Board Momemtum Data - Volatility Index and FAMA
CBVIXFAMA = pd.read_csv('/Users/ASUS/Downloads/CBVIX.csv') 

# Create dummies for policies
# TARP's Pay Czar : 2009-2010
# Dodd Frank Act : 2010 - present
# SOX : 2002 - present

# If the year is 2009-2010, the value is 1, otherwise, it is 0
secondstage['payczar'] = np.where((secondstage['YEAR'] == 2009.0) | (secondstage['YEAR'] == 2010.0), 1, 0)

secondstage = secondstage.merge(CBVIXFAMA, on = ['YEAR'], how = 'left')

# If the year is 2010-2022, the value is 1, otherwise, it is 0
secondstage['doddfrank'] = np.where((secondstage['YEAR'] == 2010.0) | (secondstage['YEAR'] == 2011.0) | 
                                    (secondstage['YEAR'] == 2012.0) |
                                    (secondstage['YEAR'] == 2013.0) | (secondstage['YEAR'] == 2014.0) | 
                                    (secondstage['YEAR'] == 2015.0) | (secondstage['YEAR'] == 2016.0) |
                                    (secondstage['YEAR'] == 2017.0) | (secondstage['YEAR'] == 2018.0) |
                                    (secondstage['YEAR'] == 2019.0) | (secondstage['YEAR'] == 2020.0) |
                                    (secondstage['YEAR'] == 2021.0) | (secondstage['YEAR'] == 2022.0), 1, 0) 


### STATA CODE ###
clear

cd "C:\Users\ASUS\Downloads"

* Compensation - Total SEC - With Policy Dummies and Volatility Index *
ssc install estout, replace

import delimited "stockvolfinal1.csv"

** First Stage **
reg adjusted_tc emp tenure payczar doddfrank cbvix mom, vce(robust) // First Stage IV 

est store reg1

predict ajtc_hat, xb // resid 

reg adjusted_stockoptions emp tenure payczar doddfrank cbvix mom, vce(robust) // First Stage IV 

test emp tenure

est store reg2

predict ajso_hat, xb // resid 


** Second Stage **
generate lnasd = ln(annualized_sd)

reg lnasd ajtc_hat ajso_hat payczar doddfrank cbvix mom, vce(robust) // Second Stage IV

test payczar doddfrank

test cbvix mom

test payczar doddfrank cbvix mom

est store reg_2SLS

esttab reg1 reg2 reg_2SLS, b(3) se(3) star compress nogap s(N r2)

estimates table, star(.1 .05 .01)
