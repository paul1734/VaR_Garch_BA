# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 10:37:47 2021

@author: paulk
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearmodels import PanelOLS
from scipy import stats
from scipy import optimize
import statsmodels.api as sm
from statsmodels.formula.api import ols
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
import scipy.optimize as opt
import statsmodels.formula.api as smf

##################
# DOWNLOAD CURRENCIES & STOCK PRICES; CLEAN DATA
##################
currencies = ['EURUSD=X','JPYUSD=X','ILSUSD=X','HKDUSD=X',
              'DKKUSD=X','CADUSD=X','AUDUSD=X','SEKUSD=X',
              'NOKUSD=X','KRWUSD=X','GBPUSD=X','INRUSD=X',
              'TWDUSD=X','BRLUSD=X','CNYUSD=X']

currencies_yf_download = yf.download(currencies, start = "2007-01-01", \
    end = "2020-01-01", period = "1d").Close
currencies_yf = currencies_yf_download.copy()
currencies_yf.isnull().values.any()
currencies_yf = currencies_yf.fillna(method='ffill')
currencies_yf = currencies_yf.bfill(limit=3).ffill(limit=3) 
currencies_yf.isnull().sum()
currency_test = currencies_yf.copy()

#seems to be fixed
currencies_yf.at['2011-10-25 00:00:00', 'TWDUSD=X'] = 0.0340
currencies_yf.at['2014-12-31 00:00:00', 'TWDUSD=X'] = 0.0324

ticker_energy_all =[
'XOM','CVX','TTE','BP','RDS-A','RDS-B','COP','ENB','SLB',
'TRP','EOG','PSX','MPC','CNQ','VLO','SU','KMI','PXD','ENI.MI','WMB','NESTE.HE',
'OXY','OKE','HAL','REPYY','EQNR','WPL.AX','HES','BKR','LNG','PBA','5020.T',
'DVN','STO.AX','CVE','IPXHY','COG','OMV.VI','OSH.AX','CCJ','ORG.AX','LUNE.ST','IPL.TO','GALP.LS',
'IMO','5019.T','TS','ALD.AX','PKI.TO','KEY.TO','VPK.AS','SOL.AX',"PXT.TO","BCEI","MGY","WLL",
"CPG","MEG.TO","ERF","BEPTF","VET","KOS","DECPF","CNX","SM","WCP.TO","CPE","EPD","CNE.L","TLW.L","ENOG.L","1662.T",
'VWS.CO','ORSTED.CO','ENPH','SEDG','SGRE.MC','RUN','GCPEF','FSLR',
'NPI.TO','VER.VI','NEP','ORA','SCATC.OL','SPWR','REGI',
'ELI.BR','BLX.TO','NOVA','INE.TO','CWEN','ECV.DE','NDX1.DE','ERG.MI','RNW.TO','CVA',
'TPIC','ENLT.TA','SLR.MC','PCELL.ST','ABIO.PA','9519.T','VBK.DE',
'CWEN-A','FKR.MI','ENRG.TA','TENERGY.AT','REX','FF','0182.HK','9517.T','PNE.TO','SIFG.AS'
,'GGI.F',"FCEL","AZRE","ADX","336260.KS","CSIQ","AVA","ALE","NEL.OL","GPRE","SXI","DRX.L","S92.DE",
"ADANIGREEN.NS","0257.HK","JKS","112610.KS","3576.TW","OMGE3.SA","1798.HK","0658.HK","DYN.F",
"BE","CIG","AY","002506.SZ","HE","BLDP","IDA","3868.HK"]

all_energy_download = yf.download(ticker_energy_all, start = "2007-01-01", end = "2020-01-01", period = "1d").Close
all_energy = all_energy_download.copy()

# convert FX stocks into USD; clean data
currencies_yf = all_energy.merge(currencies_yf, left_index=True, \
    right_index=True, how='outer')
currencies_yf = currencies_yf[currencies]
all_energy = all_energy.merge(currencies_yf, left_index=True, \
    right_index=True, how='outer')
all_energy = all_energy[ticker_energy_all]
all_energy_index = all_energy.index
all_energy_index.equals(currencies_yf.index)
currencies_yf = currencies_yf.fillna(method='ffill')
currencies_yf = currencies_yf.bfill(limit=3).ffill(limit=3) 
currencies_yf.isnull().sum()
# fill na values by former and latter values
all_energy.isnull().sum()
all_energy = all_energy.bfill(limit=3).ffill(limit=3) 
test_merge = all_energy.merge(currencies_yf, left_index=True, \
    right_index=True, how='outer')
test_merge = test_merge.bfill(limit=3).ffill(limit=3) 
test_merge.isnull().values.any()
test_merge2 = test_merge.copy()
list_convcompEUR = ['ENI.MI','NESTE.HE','OMV.VI','GALP.LS','VPK.AS',
                    'SGRE.MC','VER.VI','ELI.BR', 'ECV.DE','NDX1.DE','ERG.MI',
                    'SLR.MC','ABIO.PA','FKR.MI','TENERGY.AT','SIFG.AS','GGI.F',
                    'S92.DE','DYN.F']
list_convcompILS = ['ENRG.TA','ENLT.TA']
list_convcompJPY = ['5020.T','5019.T','1662.T','9517.T','9519.T']
list_convcompAUD = ['SOL.AX','ALD.AX','ORG.AX','OSH.AX','STO.AX','WPL.AX']
list_convcompCAN = ['KEY.TO','PKI.TO','IPL.TO','WCP.TO',
'MEG.TO','PXT.TO','PNE.TO','RNW.TO','BLX.TO' ,'NPI.TO','INE.TO']
list_convcompGB = ['CNE.L','TLW.L','ENOG.L','DRX.L']
list_convcompHKD = ['0182.HK','0257.HK','1798.HK','0658.HK','3868.HK']

for comps in list_convcompEUR:
    test_merge2[comps] = test_merge2[comps].multiply(test_merge2['EURUSD=X'], axis="index")
for comps in list_convcompILS:
    test_merge2[comps] = test_merge2[comps].multiply(test_merge2['ILSUSD=X'], axis="index")
for comps in list_convcompJPY:
    test_merge2[comps] = test_merge2[comps].multiply(test_merge2['JPYUSD=X'], axis="index")
for comps in list_convcompAUD:
    test_merge2[comps] = test_merge2[comps].multiply(test_merge2['AUDUSD=X'], axis="index")
for comps in list_convcompCAN:
    test_merge2[comps] = test_merge2[comps].multiply(test_merge2['CADUSD=X'], axis="index")
for comps in list_convcompGB:
    test_merge2[comps] = test_merge2[comps].multiply(test_merge2['GBPUSD=X'], axis="index")
for comps in list_convcompHKD:
    test_merge2[comps] = test_merge2[comps].multiply(test_merge2['HKDUSD=X'], axis="index")
test_merge2['LUNE.ST'] = test_merge2[['LUNE.ST']].multiply(test_merge2['SEKUSD=X'], axis="index")
test_merge2['PCELL.ST'] = test_merge2[['PCELL.ST']].multiply(test_merge2['SEKUSD=X'], axis="index")
test_merge2['3576.TW'] = test_merge2[['3576.TW']].multiply(test_merge2['TWDUSD=X'], axis="index")
test_merge2['OMGE3.SA'] = test_merge2[['OMGE3.SA']].multiply(test_merge2['BRLUSD=X'], axis="index")
test_merge2['002506.SZ'] = test_merge2[['002506.SZ']].multiply(test_merge2['CNYUSD=X'], axis="index")
test_merge2['ADANIGREEN.NS'] = test_merge2[['ADANIGREEN.NS']].multiply(test_merge2['INRUSD=X'], axis="index")
test_merge2['336260.KS'] = test_merge2[['336260.KS']].multiply(test_merge2['KRWUSD=X'], axis="index")
test_merge2['112610.KS'] = test_merge2[['112610.KS']].multiply(test_merge2['KRWUSD=X'], axis="index")
test_merge2['ORSTED.CO'] = test_merge2[['ORSTED.CO']].multiply(test_merge2['DKKUSD=X'], axis="index")
test_merge2['SCATC.OL'] = test_merge2[['SCATC.OL']].multiply(test_merge2['NOKUSD=X'], axis="index")
test_merge2['NEL.OL'] = test_merge2[['NEL.OL']].multiply(test_merge2['NOKUSD=X'], axis="index")

currencies_yf.isnull().values.any()
all_usd = test_merge2[ticker_energy_all]

# Data Cleaning, First Log Diff, Adfuller Stationarity Test+
list_usd_shift = []
for i in all_usd:    
    list_append = np.log(all_usd[i]/all_usd[i].shift(1)).dropna()
    list_usd_shift.append(list_append)

log_shift_df = pd.DataFrame(list_usd_shift)
log_shift_df_T = log_shift_df.transpose()
""" 87 companies with full time series """
log_nonan = log_shift_df_T[log_shift_df_T.columns[~log_shift_df_T.isnull().any()]]
log_nonan_100 = log_nonan*100
log_nonan_100 = log_nonan*100
# 57 companies with different length
# 58 companies just with NaN values; 58 + 86 (noNaN companies) = 144 i 
log_justnan = log_shift_df_T[log_shift_df_T.columns[~log_shift_df_T.notnull()\
.all()]]
log_justnan_100 = log_justnan*100
# data cleaning of 57
# GGI.F has very spotty time series, is of no use
log_justnan_100 = log_justnan_100.drop(columns=['GGI.F'])
log_justnan_100_c = log_justnan_100.copy()
log_justnan_100_listnan =[]
for column in log_justnan_100_c:
    droppedlognan_c = log_justnan_100_c.dropna(axis=0, subset=[column])
    log_justnan_100_listnan.append(droppedlognan_c[column])
copy_list57 = log_justnan_100_listnan.copy()


# ADFULLER
# test for stationarity 86 full time series
log_nonan_100_list = log_nonan_100.transpose().values.tolist()
adfuller_pvalues86 = []
for returns in range(len(log_nonan_100_list)):
    adfuller_result = adfuller(log_nonan_100_list[returns])
    adfuller_pvalues86.append(adfuller_result)
    
# all are significantly stationary, 
# p-value for every time series below 0.05
adfuller_pvalues57 = []
for returns in range(len(copy_list57)):
    adfuller_result = adfuller(copy_list57[returns])
    adfuller_pvalues57.append(adfuller_result)
# non significant adfuller test: 336260.KS
log_justnan_100_c = log_justnan_100_c.drop('336260.KS', axis=1)
# WEEKLY DATA # good 86
log_nonan_100_weeklyasfreq = log_nonan_100.asfreq("W-FRI")
# bad 57
log_justnan_100_c_weeklyasfreq = log_justnan_100_c.asfreq("W-FRI")
# delete NaN again for every column in the 57 for every
logjustnan_weekly =[]
for column in log_justnan_100_c_weeklyasfreq:
    droppedlognan_c = log_justnan_100_c_weeklyasfreq.dropna(axis=0, subset=[column])
    logjustnan_weekly.append(droppedlognan_c[column])
# convert list of series into 57 dataframes with their respective length
copy_list56_weekly = logjustnan_weekly.copy()

##############################################
# Question 2: Which energy type is less risky?
# VaR T-test Analysis
# Calculate GARCH VaR for every stock
# Build Regression stack; T-test of Energy Type dummy on VaR
##############################################
# Models GARCH & COND Volatility
def GARCH11_test(param, ret):
    # param is collection of the 3 parameters
    omega, alpha, beta = param
    n = len(ret)
    s = np.ones(n)
    for i in range(1,n):
        s[i] = omega + alpha*ret[i-1]**2 + beta*(s[i-1]) 
    # negative loglik for fmin (only minimization possible)
    logLik = -( (-0.5*np.log(2*np.pi)-0.5*np.log(s) - 0.5*( ret**2/s)).sum() )            
    return logLik

guess3 = (0.3,0.4,0.5)

def cond_var (param, ret):
    omega, alpha, beta = param
    variance = np.ones(len(ret))
    for t in range(1, len(ret)):
        # Simulate the variance (sigma squared)
        variance[t] = omega + alpha * ret[t-1]**2 + beta * variance[t-1]
    return variance

# Estimate GARCH Coefficients for both datasets
garch_list_82weeklyCON=[]
for firm_returns in log_nonan_100_weeklyasfreq:
    garch_step1NO = opt.minimize(GARCH11_test, guess3, \
        args=(log_nonan_100_weeklyasfreq[firm_returns]), \
        method='L-BFGS-B',bounds=((1e-10, 1), (1e-10, 1), (1e-10, 1)))
    garch_list_82weeklyCON.append(garch_step1NO)

garch_list_57weeklyCON=[]
for firm_returns in range(len(copy_list56_weekly)):
    garch_step1NO2 = opt.minimize(GARCH11_test, guess3, 
                    args=(copy_list56_weekly[firm_returns]), 
                    method='L-BFGS-B',bounds=((1e-10, 1), 
                    (1e-10, 1), (1e-10, 1)))
    garch_list_57weeklyCON.append(garch_step1NO2)
# get parameters from list
list_garch_86 = []
for item in range(len(garch_list_82weeklyCON)):
    variable_garch = garch_list_82weeklyCON[item]['x']
    list_garch_86.append(variable_garch)

list_garch_57 = []
for item in range(len(garch_list_57weeklyCON)):
    variable_garch2 = garch_list_57weeklyCON[item]['x']
    list_garch_57.append(variable_garch2)

# Use Parameters for VaR Estimation for Both datasets
condvar_82_good = []
for column in log_nonan_100_weeklyasfreq:
    for x in range (len(list_garch_86)):
        all_condvar = cond_var(list_garch_86[x], log_nonan_100_weeklyasfreq[column])
    condvar_82_good.append(all_condvar)
    
condvar_56_good = []
for x in range (len(list_garch_57)):
    all_condvar2 = cond_var(list_garch_57[x], copy_list56_weekly[x])
    condvar_56_good.append(all_condvar2)

# Create VaR Stack
condvar_82_df = pd.DataFrame(condvar_82_good)
condvol_82_df = np.sqrt(condvar_82_df)
condvar_56_df = pd.DataFrame(condvar_56_good)
condvol_56_df = np.sqrt(condvar_56_df)

condvol_82_dftr = condvol_82_df.transpose()
condvol_56_dftr = condvol_56_df.transpose()
names_82 = log_nonan_100_weeklyasfreq.columns
datetime_weekly82 = log_nonan_100_weeklyasfreq.index
names_56 = log_justnan_100_c_weeklyasfreq.columns
datetime_weekly56 = log_justnan_100_c_weeklyasfreq.index
# set index for conditional volatility
condvol_82_dftr.columns = names_82
condvol_82_dftr = condvol_82_dftr.set_index(datetime_weekly82)
condvol_56_dftr.columns = names_56

index_56_list = []
for ind in range(len(copy_list56_weekly)):
    index_56 = copy_list56_weekly[ind].index
    index_56_list.append(index_56)
    
index_56_df = pd.DataFrame(index_56_list)
index_56_df.index = names_56
index_56_df = index_56_df.transpose()
#########################
# VOLA WRONG
# set index for each series accordingly, log just nan as quarterly
log_justnan_100_c_quarterly = log_justnan_100_c_weeklyasfreq.asfreq('Q')

index_56_dft = index_56_df.transpose().stack()
condvol_56_st = condvol_56_dftr.transpose().stack()
index_56_dft = index_56_dft.reset_index()
condvol_56_st = condvol_56_st.reset_index()
index_56_dft.columns = ["Company Name Ind", "Number per Series","Datetime"]
condvol_56_st.columns = ["Company Name", "Number per Series", "Cond Vol"]
CondVol56_stack = pd.concat([index_56_dft,condvol_56_st], join='inner', axis=1)
CondVol56_stack = CondVol56_stack.set_index("Company Name")
CondVol56_stack = CondVol56_stack.drop("Company Name Ind", axis=1)
CondVol56_stack = CondVol56_stack.drop("Number per Series", axis=1)

CondVol56_df = pd.DataFrame(CondVol56_stack)
CondVol56_df = CondVol56_df.reset_index()
CondVol56_df = CondVol56_df.set_index(["Company Name","Datetime"])
CondVol56_df_weekly = CondVol56_df.copy()
CondVol56_df = CondVol56_df.groupby([pd.Grouper(level='Company Name'),  
            pd.Grouper(level='Datetime', freq='Q')]).mean()

CVOL_57stack_noindex = CondVol56_df.reset_index()
CVOL_57stack_noindex.columns = ["Name" , "Date", "CondVol"]
CVOL_57stack_noindex["VaR"] = CVOL_57stack_noindex["CondVol"]*(-1.960)

condvol_82_dftr_QS = condvol_82_dftr.resample('QS').mean()
# get stack of cond vola, multiindex: time and company name
CVOL_82stack = condvol_82_dftr_QS.transpose().stack()
CVOL_82stack_noindex = CVOL_82stack.reset_index()
CVOL_82stack_noindex.columns = ["Name" , "Date", "CondVol"]
CVOL_82stack_noindex["VaR"] = CVOL_82stack_noindex["CondVol"]*(-1.960)

# Delete the bad VaR estimates due to length and bad GARCH
CVOL_57stack_noindex = CVOL_57stack_noindex.set_index("Name")
CVOL_57stack_noindex = CVOL_57stack_noindex\
    .drop(['DECPF', 'ENOG.L','GCPEF','REGI','NOVA','PCELL.ST','FF',\
           'SIFG.AS','MGY','9519.T','ADANIGREEN.NS','OMGE3.SA',\
           'BE', '3868.HK'],axis=0)
CVOL_57stack_noindex = CVOL_57stack_noindex.reset_index()

CVOL_57stack_noindex.columns = ["Name" , "Date", "CondVol", "VaR"]
CVOL_57stack_dateindex = pd.DatetimeIndex(CVOL_57stack_noindex["Date"]) + pd.DateOffset(1)
CVOL_57stack_dateindex = pd.DataFrame(CVOL_57stack_dateindex)
CVOL_57stack_noindex["Date"] = CVOL_57stack_dateindex["Date"]
all_usd_stack_Quart =  pd.concat([CVOL_82stack_noindex,\
                            CVOL_57stack_noindex])
    
########
# Weekly VaR Estimate for Fossil and Green Energy
########
condvol_82_dftr
CondVol56_df_weekly
# turn stack into table
CondVol56_df_weekly_pt = CondVol56_df_weekly.pivot_table(index='Datetime',columns='Company Name',\
                     values='Cond Vol',aggfunc='mean') 
# drop companies with bad length and bad garch
CondVol56_df
CondVol56_df_weekly_pt = CondVol56_df_weekly_pt\
    .drop(['DECPF', 'ENOG.L','GCPEF','REGI','NOVA','PCELL.ST','FF',\
           'SIFG.AS','MGY','9519.T','ADANIGREEN.NS','OMGE3.SA',\
           'BE', '3868.HK'],axis=1)
CondVol_ptmerge = pd.concat([condvol_82_dftr,CondVol56_df_weekly_pt], axis=1)
VaR_weekly = CondVol_ptmerge*(-1.960)

# Create FOSSIL / ALT DUMMIES
FOS = {'XOM', 'CVX', 'TTE', 'BP','RDS-A', 'RDS-B', 'COP', 'ENB', 'SLB', 'TRP', 
       'EOG', 'CNQ','VLO','SU', 'PXD', 'ENI.MI', 'WMB', 'NESTE.HE', 'OXY', 
       'OKE', 'HAL', 'REPYY',
        'EQNR', 'WPL.AX', 'HES', 'BKR', 'LNG', '5020.T', 'DVN', 'STO.AX', 'COG',
        'OMV.VI', 'OSH.AX', 'CCJ', 'ORG.AX', 'LUNE.ST', 'IPL.TO', 'GALP.LS',
        'IMO', '5019.T', 'TS', 'ALD.AX', 'PKI.TO', 'KEY.TO', 'VPK.AS', 'SOL.AX',
       'WLL', 'CPG', 'ERF', 'CNX', 'SM', 'CPE', 'EPD','TLW.L', '1662.T',\
       'PSX', 'MPC', 'KMI', 'PBA', 'CVE', 'IPXHY',
       'PXT.TO', 'BCEI', 'MGY', 'MEG.TO', 'BEPTF', 'VET', 'KOS', 'DECPF',
       'WCP.TO', 'CNE.L', 'ENOG.L', 'PNE.TO'}

ALT = {'VWS.CO','SGRE.MC', 'FSLR', 'NPI.TO', 'VER.VI', 'ORA', 'SPWR', 'ELI.BR',
       'BLX.TO', 'NDX1.DE', 'ERG.MI', 'CVA', 'ENLT.TA', 'ABIO.PA', 'VBK.DE',
       'FKR.MI', 'REX', 'FCEL', 'ADX', 'CSIQ', 'AVA', 'ALE', 'NEL.OL', 'GPRE',
       'SXI', 'DRX.L', '0257.HK', 'CIG', 'HE', 'BLDP', 'IDA',\
        'ORSTED.CO', 'ENPH', 'SEDG', 'RUN',
       'GCPEF', 'NEP', 'SCATC.OL', 'REGI', 'NOVA', 'INE.TO', 'CWEN',
       'ECV.DE', 'RNW.TO', 'TPIC', 'SLR.MC', 'PCELL.ST', '9519.T',
       'CWEN-A', 'ENRG.TA', 'TENERGY.AT', 'FF', '0182.HK', '9517.T',
       'SIFG.AS','AZRE', 'S92.DE', 'ADANIGREEN.NS', 'JKS','112610.KS', 
       '3576.TW', 'OMGE3.SA', '1798.HK', '0658.HK', 'DYN.F',
       'BE', 'AY', '002506.SZ', '3868.HK'}

all_usd_stack_Quart['Energy Type'] = np.select([all_usd_stack_Quart['Name']\
                                    .isin(FOS),all_usd_stack_Quart['Name']\
                                    .isin(ALT)],['Fossil','Alternative'])
all_usd_stack_Quart['Energy Type'] = pd.get_dummies(all_usd_stack_Quart['Energy Type'])

VaR_weekly_st = VaR_weekly.transpose().stack()
VaR_weekly_st = VaR_weekly_st.reset_index()
VaR_weekly_st.columns = ["Name", "Date","VaR"]
VaR_weekly_st['Energy Type'] = np.select([VaR_weekly_st['Name']\
                                    .isin(FOS),VaR_weekly_st['Name']\
                                    .isin(ALT)],['Fossil','Alternative'])
VaR_weekly_st['Energy Type'] = pd.get_dummies(VaR_weekly_st['Energy Type'])
#######################################################
# Create Factors
# Momentum, Dividends, Price/earnings, Market Cap
########################################################
#########################
# Question 1: Are energy stocks affected by factors?
# Linear Mixed Effects Models
#########################
# MOMENTUM FACTOR 
# Calculate Returns 
################
all_usd_stack_companies = all_usd_stack_Quart['Name'].unique()
all_usd_stack_companies_list = list(all_usd_stack_companies)

all_usd128 = all_usd[all_usd_stack_companies_list]
returns_data = all_usd128.copy()
returns_data = returns_data.resample("M").mean()
returns_data = returns_data.apply(func = lambda x: x.shift(-1)/x - 1, axis = 0)
returns_data = returns_data*100
returns_dataQ = returns_data.resample("QS").mean()

factor_signal = all_usd128.copy() # resample with monthly mean
momentum_signal = factor_signal.resample("M").mean()
momentum_signal = momentum_signal.apply(func = lambda x: x.shift(1)/\
                          x.shift(12) - 1, axis = 0)
#momentum_signal = momentum_signal.apply(func = lambda x: x.shift(1)
#                            , axis = 0)
momentum_signal = momentum_signal.rank(axis = 1, pct=True, numeric_only=True)
for col in momentum_signal.columns:
    momentum_signal[col] = np.where(momentum_signal[col] >= 0.70, 1,\
                           np.where(momentum_signal[col] < 0.30, -1,0))

# add to MVaRQuart
momentum_signalQ = momentum_signal.asfreq('Q')
returns_signals = np.multiply(returns_data, momentum_signal)
returns_signals_quart = returns_signals.asfreq('Q')
# get lag of momentum
# returns_signals_quart_lag = returns_signals_quart.copy()
# returns_signals_quart_lag = returns_signals_quart_lag.shift(1)
# returns_signals_quart_lag = returns_signals_quart_lag.dropna(how="all")
#
# NOW USE returns_signals_quart_lag
# returns_signals_quart = returns_signals.asfreq("Q")
ret_sig_quart_stack = returns_signals_quart.transpose().stack()
ret_sig_quart_stack = ret_sig_quart_stack.reset_index()
ret_sig_quart_stack.columns = ['Company MOM', 'Date MOM','Momentum']
ret_sig_quart_stack["Date MOM"] = pd.DatetimeIndex(ret_sig_quart_stack['Date MOM']) + pd.DateOffset(1)

######
MRVaR_QR = all_usd_stack_Quart.copy()
MRVaR_QR.columns = ['Company Ticker','Quarter','CondVol','VaR',\
                    'Energy Type']
####
MVaR_Mom = MRVaR_QR.merge(ret_sig_quart_stack,how='outer',\
    left_on=['Company Ticker','Quarter'],right_on=['Company MOM',"Date MOM"])
MRVaR_QR = MRVaR_QR.merge(ret_sig_quart_stack,how='outer',\
    left_on=['Company Ticker','Quarter'],right_on=['Company MOM',"Date MOM"])
MRVaR_QR = MRVaR_QR.drop(['Date MOM', "Company MOM"],axis=1)
MVaR_Mom = MVaR_Mom.drop(['Date MOM', "Company MOM"],axis=1)
MVaR_Mom = MVaR_Mom.dropna()
###############
# IMPORT P/E Ratio, Dividends and Shares from Morningsstar as csv
data_factors = P_ETableTestCleaningcsv
data_factors.columns = data_factors.iloc[0]
data_factors = data_factors.drop(0)

##################
# VALUE: P/E Ratio
##################
peratio = data_factors.iloc[:,0:11]
peratio = peratio.set_index('Ticker PE')
peratio.columns = ["2011", "2012", "2013", "2014", "2015", "2016", "2017", \
                  "2018", "2019","2020"]
peratio = peratio.transpose()
peratio.index = pd.to_datetime(peratio.index)
peratio = peratio.resample('QS').ffill()
peratio = peratio.reset_index()
peratio.rename( columns={'index' :'Date'}, inplace=True )
peratio = peratio.set_index('Date')
peratio = peratio.iloc[:36,:]
peratio = peratio.apply(func = lambda x: x.shift(4), axis = 0)

pe_signal = peratio.rank(axis = 1, pct=True, numeric_only=True)
for col in pe_signal.columns:
    pe_signal[col] = np.where(pe_signal[col] >= 0.70, -1,\
                           np.where(pe_signal[col] < 0.30, 1,0))

returns_dataQ_2011 = returns_dataQ.iloc[16:,:]
pereturns_signals = np.multiply(returns_dataQ_2011, pe_signal)
pereturns_signals_quart = pereturns_signals.resample('QS').mean()
pereturns_signals_quart_stack = pereturns_signals_quart.transpose().stack()
pereturns_signals_quart_stack = pereturns_signals_quart_stack.reset_index()
pereturns_signals_quart_stack.columns = ['Company PE', 'Date PE','PE']
PE_Q_Stack = pereturns_signals_quart_stack.copy()

VaRPE = MVaR_Mom.merge(PE_Q_Stack,how='outer',left_on=['Company Ticker',\
                                'Quarter'],right_on=['Company PE','Date PE'])
Regression_all = MRVaR_QR.merge(PE_Q_Stack,how='outer',\
	left_on=['Company Ticker','Quarter'],right_on=['Company PE','Date PE'])
VaRPE = VaRPE.drop(["Company PE", "Date PE"], axis=1)
Regression_all = Regression_all.drop(["Company PE", "Date PE"], axis=1)
###################
# SIZE: Market Cap
###################
shares = data_factors.iloc[:,24:35]
shares = shares.set_index('Ticker SH')
shares.columns = ["2011", "2012", "2013", "2014", "2015", "2016", "2017", \
                  "2018", "2019", "2020"]
shares = shares.transpose()
shares.index = pd.to_datetime(shares.index)
shares = shares.resample('QS').ffill()
shares = shares.iloc[:36,:]
all_usd128Q = all_usd128.resample("QS").mean()
all_usd128Q_2011 = all_usd128Q.iloc[16:,:]
marketcap = all_usd128Q_2011*shares
marketcap_signal = marketcap
for col in marketcap.columns:
    marketcap_signal[col] = np.where(marketcap[col] >= 10000.00, -1,\
                           np.where(marketcap[col] < 10000.00, 1,0))

returns_dataQ_2011 = returns_dataQ.iloc[16:,:]
mc_returns_signals = np.multiply(returns_dataQ_2011, marketcap_signal)
mc_returns_signals_Q = mc_returns_signals.resample('QS').mean()
mc_returns_signals_Q_st = mc_returns_signals_Q.\
   transpose().stack()
mc_returns_signals_Q_st = mc_returns_signals_Q_st.reset_index()
mc_returns_signals_Q_st.columns = ['Company MC', 'Date MC','Market Cap']
MC_Q_Stack = mc_returns_signals_Q_st.copy()
VaRPE = VaRPE.merge(MC_Q_Stack,how='outer',left_on=['Company Ticker',\
                                'Quarter'],right_on=['Company MC','Date MC'])
VaRPE = VaRPE.drop(["Company MC", "Date MC"], axis=1)

VaRPE = VaRPE.merge(ret_sig_quart_stack,how='outer',\
    left_on=['Company Ticker','Quarter'],right_on=['Company MOM',"Date MOM"])
Regression_all = Regression_all.merge(MC_Q_Stack,how='outer',\
    left_on=['Company Ticker','Quarter'],right_on=['Company MC',"Date MC"])
VaRPE = VaRPE.drop(["Company MOM", "Date MOM"], axis=1)
Regression_all = Regression_all.drop(["Company MC", "Date MC"], axis=1)
#################################
# DIVIDENDS
# dividends need to be converted tu USD; convert with mean FX rate of each year
# from 2011 until 2019
##################################
currencies_yf_A = currencies_yf.resample("A").mean()
currencies_yf_A2011 = currencies_yf_A.iloc[3:,:]
dividends = data_factors.iloc[:,12:23]
dividends = dividends.set_index('Ticker DIV')
dividends.columns = ["2011", "2012", "2013", "2014", "2015", "2016", "2017", \
                  "2018", "2019", "2020"]
dividends = dividends.transpose()
dividends.index = pd.to_datetime(dividends.index)
dividends = dividends.set_index(currencies_yf_A2011.index)
dividends = dividends.merge(currencies_yf_A2011, left_index=True, \
    right_index=True, how='outer')

Cad = ["ENB", "TRP", "CNQ", "SU", "CCJ","IMO", "IPL.TO", "PKI.TO","KEY.TO",  
		"CPG", "ERF", "NPI.TO", "BLX.TO", "PBA", "CVE", "MEG.TO", "VET", "WCP.TO",
		"INE.TO", "RNW.TO", "PNE.TO"]
Jpy = ["5020.T", "5019.T", "1662.T", "IPXHY", "9517.T"]
Cny = ["0182.HK", "JKS", "1798.HK", "0658.HK", "DYN.F", "002506.SZ"]
Aud = ["ORG.AX","ALD.AX", "SOL.AX", "BEPTF"]
Ils = ["ENLT.TA", "ENRG.TA"]
Eur = ["ENI.MI", "NESTE.HE", "REPYY", "OMV.VI", "GALP.LS","VPK.AS","VWS.CO","SGRE.MC",
		"VER.VI", "ELI.BR", "NDX1.DE", "ERG.MI", "ABIO.PA", "VBK.DE","FKR.MI","ECV.DE",
		"SLR.MC", "TENERGY.AT", "S92.DE"]
Nok = ["NEL.OL", "SCATC.OL"]
Krw = ["112610.KS"]
Dkk = ["ORSTED.CO"]
Brl =["CIG"]
Twd = ["3576.TW"]
Inr = ["AZRE"]
Gbp = ["DRX.L"]
Hkd = ["0257.HK"]
for divid in Eur:
    dividends[divid] = dividends[divid].multiply(dividends['EURUSD=X'], \
    	axis="index")
for divid in Aud:
    dividends[divid] = dividends[divid].multiply(dividends['AUDUSD=X'], \
    	axis="index")
for divid in Cad:
    dividends[divid] = dividends[divid].multiply(dividends['CADUSD=X'], \
    	axis="index")
for divid in Jpy:
    dividends[divid] = dividends[divid].multiply(dividends['JPYUSD=X'], \
    	axis="index")
for divid in Cny:
    dividends[divid] = dividends[divid].multiply(dividends['CNYUSD=X'], \
    	axis="index")
for divid in Ils:
    dividends[divid] = dividends[divid].multiply(dividends['ILSUSD=X'], \
    	axis="index")
for divid in Nok:
    dividends[divid] = dividends[divid].multiply(dividends['NOKUSD=X'], \
    	axis="index")
    
dividends["112610.KS"] = dividends[["112610.KS"]].multiply(dividends['KRWUSD=X'], axis="index")
dividends["ORSTED.CO"] = dividends[["ORSTED.CO"]].multiply(dividends['DKKUSD=X'], axis="index")
dividends["CIG"] = dividends[["CIG"]].multiply(dividends['BRLUSD=X'], axis="index")
dividends['3576.TW'] = dividends[['3576.TW']].multiply(dividends['TWDUSD=X'], axis="index")
dividends["AZRE"] = dividends[["AZRE"]].multiply(dividends['INRUSD=X'], axis="index")
dividends["DRX.L"] = dividends[["DRX.L"]].multiply(dividends['GBPUSD=X'], axis="index")
dividends["0257.HK"] = dividends[["0257.HK"]].multiply(dividends['HKDUSD=X'], axis="index")

dividends = dividends.iloc[:,:128]
dividends_stat = dividends.copy()
dividends = dividends.resample('QS').ffill()
dividends = dividends.iloc[1:,:]

# create factor
dividends = dividends.apply(func = lambda x: x.shift(4), axis = 0)
dividends_signal = dividends.rank(axis = 1, pct=True, numeric_only=True)
for col in dividends_signal.columns:
    dividends_signal[col] = np.where(dividends_signal[col] >= 0.70, 1,\
                           np.where(dividends_signal[col] < 0.30, -1,0))

dividends_signal = dividends_signal.iloc[1:,:]
returns_dataQ_2011 = returns_dataQ.iloc[17:,:]
dividends_signals = np.multiply(returns_dataQ_2011, dividends_signal)
dividends_signals_Q = dividends_signals.resample('QS').mean()

dividends_signals_Q_st = dividends_signals_Q.transpose().stack()
dividends_signals_Q_st = dividends_signals_Q_st.reset_index()
dividends_signals_Q_st.columns = ['Ticker DIV', 'Date DIV','Dividends']

Regression_all_test = Regression_all.copy()
Regression_all_test = Regression_all_test.merge(dividends_signals_Q_st,how='outer',\
    left_on=['Company Ticker', 'Quarter'],right_on=['Ticker DIV','Date DIV'])
returns_stack = returns_dataQ.transpose().stack()
returns_stack = returns_stack.reset_index()
returns_stack.columns = ["Index", "Quarter","Returns"]
Regression_all_test = Regression_all_test.merge(returns_stack,how='outer',\
    left_on=['Company Ticker', 'Quarter'],right_on=['Index','Quarter'])
Regression_all_test = Regression_all_test.iloc[:5903,:]

#########
# TABLE 3 SUMMARY STATISTIC
# Descriptive Stats 3 fundamentals for fossil and alternative
peratio_stat = data_factors.iloc[:,0:11]
dividends_stat = dividends_stat.transpose()
dividends_stat = dividends_stat.reset_index()
peratio_stat['Energy Type'] = np.select([peratio_stat['Ticker PE']\
                                    .isin(FOS),peratio_stat['Ticker PE']\
                                    .isin(ALT)],['Fossil','Alternative',])
peratio_stat['Energy Type'] = pd.get_dummies(peratio_stat['Energy Type'])


dividends_stat['Energy Type'] = np.select([dividends_stat['index']\
                                    .isin(FOS),dividends_stat['index']\
                                    .isin(ALT)],['Fossil','Alternative'])
dividends_stat['Energy Type'] = pd.get_dummies(dividends_stat['Energy Type'])

peratio_stat_fossil = peratio_stat.loc[peratio_stat\
                    ['Energy Type'] == 0]
print(peratio_stat_fossil.mean())
print(peratio_stat_fossil.mean().mean())

peratio_stat_alt = peratio_stat.loc[peratio_stat\
                    ['Energy Type'] == 1]
print(peratio_stat_alt.mean())
print(peratio_stat_alt.mean().mean())

dividends_stat_fossil = dividends_stat.loc[dividends_stat\
                    ['Energy Type'] == 0]
print(dividends_stat_fossil.mean())
print(dividends_stat_fossil.mean().mean())

dividends_stat_alt = dividends_stat.loc[dividends_stat\
                    ['Energy Type'] == 1]
print(dividends_stat_alt.mean())
print(dividends_stat_alt.mean().mean())

###############
# VaR Market adjusted
# Weekly with average market cap correction
quarterly_marketcap = all_usd128Q_2011*shares
mean_marketcap = quarterly_marketcap.mean()
mean_marketcap = mean_marketcap.reset_index()
mean_marketcap.columns = ["Name", "Market Cap"]
# Weekly
mean_marketcap.at[12, 'Market Cap'] = 2266.00
mean_marketcap.at[112, 'Market Cap'] = 642.48
mean_marketcap = mean_marketcap.set_index("Name")
VaR_marketcap_test = VaR_weekly.transpose()
mean_marketcap_sort = mean_marketcap.sort_index()

mean_marketcap_sort["MC in Percent"] = (mean_marketcap_sort['Market Cap'] / \
                        mean_marketcap_sort['Market Cap'].sum()) * 100

#########################
VaR_marketcap_test_sort = VaR_marketcap_test.sort_index()
VaR_marketcap_test_sortT = VaR_marketcap_test_sort.transpose()
VaR_mcadjusted = VaR_marketcap_test_sort.multiply(mean_marketcap_sort\
                        ["MC in Percent"], axis="index")
VaR_mcadjusted_st = VaR_mcadjusted.stack()
VaR_mcadjusted_st = VaR_mcadjusted_st.reset_index()
VaR_mcadjusted_st.columns = ["Name", "Date", "VaR adjusted"]
VaR_weekly_st1 = pd.merge(VaR_mcadjusted_st,VaR_weekly_st, on=["Name",'Date'])
# add constant to VaR
VaR_weekly_st1 = sm.add_constant(VaR_weekly_st1)
#########
##############################
# RESULTS 
##############################

#####
# OLS: VaR ~ Energy Type
#####
# VaR ~ Energy Type with weekly data

reg_Regression_all = sm.OLS(endog=VaR_weekly_st1['VaR'],\
              exog=VaR_weekly_st1[["const","Energy Type"]]
,hasconst=True, missing="drop")  
reg_Regression_all_fit = reg_Regression_all.fit()
reg_Regression_all_fit = reg_Regression_all_fit.get_robustcov_results(cov_type='HAC', maxlags=1)
print(reg_Regression_all_fit.summary())

reg_Regression_all = sm.OLS(endog=VaR_weekly_st1['VaR adjusted'],\
              exog=VaR_weekly_st1[["const","Energy Type"]]
,hasconst=True, missing="drop")  
reg_Regression_all_fit = reg_Regression_all.fit()
reg_Regression_all_fit = reg_Regression_all_fit.get_robustcov_results(cov_type='HAC', maxlags=1)
print(reg_Regression_all_fit.summary())
########################################
# Divide VaR in three parts
VaR_weekly_st["Date"].unique()
VaR_weekly_3period_1st = VaR_weekly_st1\
   [VaR_weekly_st1["Date"]\
    .isin(pd.date_range('2007-01-05', '2011-5-02'))]
       
VaR_weekly_3period_2nd = VaR_weekly_st1\
   [VaR_weekly_st1["Date"]\
    .isin(pd.date_range('2011-5-02', '2015-9-02'))]
       
VaR_weekly_3period_3rd = VaR_weekly_st1\
   [VaR_weekly_st1["Date"]\
    .isin(pd.date_range('2015-9-02', '2019-12-27'))]
       
##########################
# market cap adjusted for 3 time periods
# Complete regression with endog & exog:
# both for ['VaR'] & ['VaR adjusted']
# 1) VaR_weekly_3period_1st
# 2) VaR_weekly_3period_2nd
# 3) VaR_weekly_3period_3rd

# example VaR all subperiods:
reg_Regression_all = sm.OLS(endog=VaR_weekly_3period_1st['VaR'],\
              exog=VaR_weekly_3period_1st[["const","Energy Type"]]
,hasconst=True, missing="drop")  
reg_Regression_all_fit = reg_Regression_all.fit()
reg_Regression_all_fit = reg_Regression_all_fit.get_robustcov_results(cov_type='HAC', maxlags=1)
print(reg_Regression_all_fit.summary())

reg_Regression_all = sm.OLS(endog=VaR_weekly_3period_2nd['VaR'],\
              exog=VaR_weekly_3period_2nd[["const","Energy Type"]]
,hasconst=True, missing="drop")  
reg_Regression_all_fit = reg_Regression_all.fit()
reg_Regression_all_fit = reg_Regression_all_fit.get_robustcov_results(cov_type='HAC', maxlags=1)
print(reg_Regression_all_fit.summary())

reg_Regression_all = sm.OLS(endog=VaR_weekly_3period_3rd['VaR'],\
              exog=VaR_weekly_3period_3rd[["const","Energy Type"]]
,hasconst=True, missing="drop")  
reg_Regression_all_fit = reg_Regression_all.fit()
reg_Regression_all_fit = reg_Regression_all_fit.get_robustcov_results(cov_type='HAC', maxlags=1)
print(reg_Regression_all_fit.summary())

# example VaR adjusted all subperiods:
reg_Regression_all = sm.OLS(endog=VaR_weekly_3period_1st['VaR adjusted'],\
              exog=VaR_weekly_3period_1st[["const","Energy Type"]]
,hasconst=True, missing="drop")  
reg_Regression_all_fit = reg_Regression_all.fit()
reg_Regression_all_fit = reg_Regression_all_fit.get_robustcov_results(cov_type='HAC', maxlags=1)
print(reg_Regression_all_fit.summary())

reg_Regression_all = sm.OLS(endog=VaR_weekly_3period_2nd['VaR adjusted'],\
              exog=VaR_weekly_3period_2nd[["const","Energy Type"]]
,hasconst=True, missing="drop")  
reg_Regression_all_fit = reg_Regression_all.fit()
reg_Regression_all_fit = reg_Regression_all_fit.get_robustcov_results(cov_type='HAC', maxlags=1)
print(reg_Regression_all_fit.summary())
    
reg_Regression_all = sm.OLS(endog=VaR_weekly_3period_3rd['VaR adjusted'],\
              exog=VaR_weekly_3period_3rd[["const","Energy Type"]]
,hasconst=True, missing="drop")  
reg_Regression_all_fit = reg_Regression_all.fit()
reg_Regression_all_fit = reg_Regression_all_fit.get_robustcov_results(cov_type='HAC', maxlags=1)
print(reg_Regression_all_fit.summary())

######
# VaR Index Plot: Fossil & Renewable

fossil_df_weekly = VaR_weekly_st1.loc[VaR_weekly_st1\
                    ['Energy Type'] == 0]
alternative_df_weekly = VaR_weekly_st1.loc[VaR_weekly_st1\
                    ['Energy Type'] == 1]

fossil_pt_weekly = fossil_df_weekly.pivot_table(index='Date',columns='Name',\
                     values='VaR',aggfunc='mean') 
alternative_pt_weekly = alternative_df_weekly.pivot_table(index='Date',columns='Name',\
                     values='VaR',aggfunc='mean') 

fossil_pt_weekly_adj = fossil_df_weekly.pivot_table(index='Date',columns='Name',\
                     values='VaR adjusted',aggfunc='mean') 
alternative_pt_weekly_adj = alternative_df_weekly.pivot_table(index='Date',columns='Name',\
                     values='VaR adjusted',aggfunc='mean')    
 
# Plot no adjustment
energy_plot_weekly = pd.DataFrame()
energy_plot_weekly['Mean VaR Fossil'] = fossil_pt_weekly.mean(axis=1)
energy_plot_weekly['Mean VaR Alternative'] = alternative_pt_weekly.mean(axis=1)
ax = energy_plot_weekly.plot(linewidth=2, fontsize=12);
ax.set_xlabel('Time');
ax.set_ylabel("Value-at-Risk")
ax.legend(fontsize=12);
ax.set_title('Weekly Mean Value at Risk- No Adjustment')

# Plot adjusted to market cap
energy_plot_weekly_adj = pd.DataFrame()
energy_plot_weekly_adj['Mean VaR Fossil'] = fossil_pt_weekly_adj.mean(axis=1)
energy_plot_weekly_adj['Mean VaR Alternative'] = alternative_pt_weekly_adj.mean(axis=1)
ax_adj = energy_plot_weekly_adj.plot(linewidth=2, fontsize=12);
ax_adj.set_xlabel('Time');
ax_adj.set_ylabel("Value-at-Risk")
ax_adj.legend(fontsize=12);
ax_adj.set_title('Weekly Mean Value at Risk- Market Cap Adjusted')

#######################################################
#########################
# CREATE PANEL BALANCED: To check if OLS feasible
#########################
filtered_df = Regression_all_test[Regression_all_test["Quarter"]\
            .isin(pd.date_range('2011-04-01', '2019-07-01'))]
print(filtered_df)
df_all_dropped1 = filtered_df.drop([ 'Ticker DIV', "Date DIV",
                                    "Index"], axis = 1)
df_all_dropped1["Company Ticker"].unique()
df_all_dropped1['Company Ticker'].value_counts()
df_all_dropped1["Company Ticker"]\
  [df_all_dropped1["Quarter"].isin(pd.date_range('2011-01-01','2011-04-01'))]
# companies without full quarters
# Company Ticker: 112610.KS, 9517.T , AY ,AZRE,CWEN
# ,CWEN-A, DYN.F, ENPH, NEP ,ORSTED.CO, PSX
# ,PXT.TO , RNW.TO,RUN,SEDG,TPIC, SCATC.OL

df_all_dropped2 = df_all_dropped1[~df_all_dropped1['Company Ticker'].isin(['112610.KS'])]
df_all_dropped2 = df_all_dropped2[~df_all_dropped2['Company Ticker'].isin(['002506.SZ'])]
df_all_dropped2 = df_all_dropped2[~df_all_dropped2['Company Ticker'].isin(['TENERGY.AT'])]
df_all_dropped2 = df_all_dropped2[~df_all_dropped2['Company Ticker'].isin(['9517.T'])]
df_all_dropped2 = df_all_dropped2[~df_all_dropped2['Company Ticker'].isin(['AY'])]
df_all_dropped2 = df_all_dropped2[~df_all_dropped2['Company Ticker'].isin(['AZRE'])]
df_all_dropped2 = df_all_dropped2[~df_all_dropped2['Company Ticker'].isin(['CWEN'])]
df_all_dropped2 = df_all_dropped2[~df_all_dropped2['Company Ticker'].isin(['CWEN-A'])]
df_all_dropped2 = df_all_dropped2[~df_all_dropped2['Company Ticker'].isin(['DYN.F'])]
df_all_dropped2 = df_all_dropped2[~df_all_dropped2['Company Ticker'].isin(['ENPH'])]
df_all_dropped2 = df_all_dropped2[~df_all_dropped2['Company Ticker'].isin(['NEP'])]
df_all_dropped2 = df_all_dropped2[~df_all_dropped2['Company Ticker'].isin(['ORSTED.CO'])]
df_all_dropped2 = df_all_dropped2[~df_all_dropped2['Company Ticker'].isin(['PSX'])]
df_all_dropped2 = df_all_dropped2[~df_all_dropped2['Company Ticker'].isin(['PXT.TO'])]
df_all_dropped2 = df_all_dropped2[~df_all_dropped2['Company Ticker'].isin(['RNW.TO'])]
df_all_dropped2 = df_all_dropped2[~df_all_dropped2['Company Ticker'].isin(['RUN'])]
df_all_dropped2 = df_all_dropped2[~df_all_dropped2['Company Ticker'].isin(['SEDG'])]
df_all_dropped2 = df_all_dropped2[~df_all_dropped2['Company Ticker'].isin(['TPIC'])]
df_all_dropped2 = df_all_dropped2[~df_all_dropped2['Company Ticker'].isin(['SCATC.OL'])]

# further companies outside range starting at 2011-04-01
# Company Ticker: ENRG.TA, BCEI, KOS, MPC
df_all_dropped2 = df_all_dropped2[~df_all_dropped2['Company Ticker'].isin(['ENRG.TA'])]
df_all_dropped2 = df_all_dropped2[~df_all_dropped2['Company Ticker'].isin(['BCEI'])]
df_all_dropped2 = df_all_dropped2[~df_all_dropped2['Company Ticker'].isin(['KOS'])]
df_all_dropped2 = df_all_dropped2[~df_all_dropped2['Company Ticker'].isin(['MPC'])]

#######################
# PANEL OLS
####################

df_all_dropped3 = df_all_dropped2.copy()
df_all_dropped3 = df_all_dropped3.reset_index()
df_all_dropped3 = df_all_dropped3.set_index(\
                    ["Company Ticker", "Quarter"])
df_all_dropped4 = df_all_dropped3.drop(\
                    ["CondVol", "VaR", "index", "Returns"], axis=1)
exog_all = sm.tools.tools.add_constant(df_all_dropped4)
endog = df_all_dropped3["Returns"]

model_fe = PanelOLS(endog,exog_all,entity_effects = True,\
             drop_absorbed=True)
fe_res = model_fe.fit(cov_type='clustered')
print(fe_res)
##########################################################
# Baseline spec: OLS: 
# Returns ~ factors + Energy Type ; 
# only for 2011-2019
Regression_all_test_2011 = Regression_all_test.copy()
Regression_all_test_2011 = Regression_all_test_2011\
   [Regression_all_test["Quarter"]\
    .isin(pd.date_range('2011-01-01', '2019-10-01'))]

# ALL Variables
reg_Regression_all = sm.OLS(endog=Regression_all_test_2011['Returns'],\
              exog=Regression_all_test_2011[["Energy Type",\
                  "Momentum","PE","Market Cap", "Dividends"]]
,hasconst=True, missing="drop")  
reg_Regression_all_fit = reg_Regression_all.fit()
reg_Regression_all_fit = reg_Regression_all_fit.get_robustcov_results(cov_type='HAC', maxlags=1)
print(reg_Regression_all_fit.summary())
# NO DIVIDENDS

reg_Regression_all = sm.OLS(endog=Regression_all_test_2011['Returns'],\
              exog=Regression_all_test_2011[["Energy Type",\
                  "Momentum","PE","Market Cap"]]
,hasconst=True, missing="drop")  
reg_Regression_all_fit = reg_Regression_all.fit()
reg_Regression_all_fit = reg_Regression_all_fit.get_robustcov_results(cov_type='HAC', maxlags=1)
print(reg_Regression_all_fit.summary())

# NO Energy Type

reg_Regression_all = sm.OLS(endog=Regression_all_test_2011['Returns'],\
              exog=Regression_all_test_2011[["Momentum",
                        "PE","Market Cap", "Dividends"]]
,hasconst=True, missing="drop")  
reg_Regression_all_fit = reg_Regression_all.fit()
reg_Regression_all_fit = reg_Regression_all_fit.get_robustcov_results(cov_type='HAC', maxlags=1)
print(reg_Regression_all_fit.summary())

# NO Momentum

reg_Regression_all = sm.OLS(endog=Regression_all_test_2011['Returns'],\
              exog=Regression_all_test_2011[["PE",
                        "Energy Type","Market Cap", "Dividends"]]
,hasconst=True, missing="drop")  
reg_Regression_all_fit = reg_Regression_all.fit()
reg_Regression_all_fit = reg_Regression_all_fit.get_robustcov_results(cov_type='HAC', maxlags=1)
print(reg_Regression_all_fit.summary())