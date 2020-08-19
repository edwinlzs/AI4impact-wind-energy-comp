import requests
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler,StandardScaler,OneHotEncoder
from sklearn.metrics import mean_absolute_error,mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout,Flatten,Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Concatenate


'''main model file for preprocessing and model training'''


'''Creating endpoints for data pulling'''
def add_b_suffix(og_name):
    '''creates a list of -b endpoints for data pulling'''
    return_list = []
    for f in og_name:
        return_list.append(f+"-b")
    return return_list


locations = ['guitrancourt', 'lieusaint',
             'lvs-pussay','parc-du-gatinais',
             'arville','boissy-la-riviere',
             'angerville-1','angerville-2']

wind_energy = 'energy-ile-de-france'
forecast_endpt = 'https://ai4impact.org/P003/'
analysis_endpt = 'https://ai4impact.org/P003/historical/'


'''Pulling data from the endpoints'''
model_1 = locations
model_2 = add_b_suffix(model_1)
models = [model_1, model_2]


#saves the models as CSV files
model_num = 0
for m in models:
    model_num += 1
    df = pd.read_csv(analysis_endpt+m[0]+'.csv',skiprows=3)
    df.columns = ['Time','Speed_'+m[0],'Direction_'+m[0]]
    df.set_index('Time',inplace=True)
    for i in range(1,len(m)):
        loc = m[i]
        temp = pd.read_csv(analysis_endpt+loc+'.csv',skiprows=3)
        temp.columns = ['Time','Speed_'+loc,'Direction_'+loc]
        temp.set_index('Time',inplace=True)
        df = df.merge(temp,how='left',on='Time')
        df.drop_duplicates(inplace=True)

    df.reset_index(inplace=True,drop=False)
    df.to_csv(f'model_{model_num}.csv')


#import csv files
df1 = pd.read_csv('model_1.csv')
df2 = pd.read_csv('model_2.csv')


'''download data from endpoint and save to csv'''
target = pd.read_csv('https://ai4impact.org/P003/historical/energy-ile-de-france.csv',header=None)
target.columns = ['Time','Wind Energy']
target.to_csv('target.csv',index=False)
target['Time'] = pd.to_datetime(target['Time'])
target.set_index('Time',inplace=True)


'''tansformations with OOP'''

class DataTransformer:
    def __init__(self,df):
        self.df = df
    '''Class containing methods to transform the imported data'''

    # in OOP do u not need to call return?
    def transform(self):
        '''overall transformation of data'''
        self.interpolate()
        self.add_cyclical_features()
        self.add_time_features()
        self.ohe()
        self.add_historical_windpower()
        self.add_momentum_force()
        self.scale()
        return self.df

    def interpolate(self):
        '''interpolation of data'''
        df = self.df
        df['Time'] = df['Time'].apply(lambda x : datetime.datetime.strptime(x[:-3], '%Y/%m/%d %H:%M'))
        df['Time'] = pd.to_datetime(df['Time'])    #why double the time conversion?
        df.set_index('Time',inplace=True)
        df = df.resample('1H').asfreq()    #unsure about resample and asfreq
        df.interpolate(method='cubic',axis=0,limit_direction='both',inplace=True)
        self.df = df

    def add_cyclical_features(self):
        '''converts direction into cylical inputs'''
        df = self.df
        cols = df.columns
        for c in cols:
            if 'Direction' in c:
                df[c+'_norm'] = df[c]/360
                df[c+'_sin'] = df[c+'_norm'].apply(lambda x: np.sin(x))
                df[c+'_cos'] = df[c+'_norm'].apply(lambda x: np.cos(x))
                df.drop([c,c+'_norm'],inplace=True,axis=1)

        self.df = df

    def scale(self):
        '''normalize entire dataframe'''
        df = self.df
        df = pd.DataFrame(StandardScaler().fit_transform(df),index=df.index,columns=df.columns)
        self.df = df

    def add_time_features(self):
        '''create time inputs as attributes?'''
        df = self.df
        df.reset_index(inplace=True,drop=False)
        #this is assigment of attribute?
        df['hour'] = df['Time'].apply(lambda x: x.hour).astype(str)
        df['month'] = df['Time'].apply(lambda x: x.month).astype(str)
        # df['day'] = df['Time'].apply(lambda x: x.day).astype(str)
        df.set_index('Time',inplace=True)
        self.df = df

    def ohe(self):
        '''One hot encoding of time data'''
        #what is this? I assume it standings for one hot encoding
        #doesn't it affect the entire frame vs just the select month or year?
        df = self.df
        df = pd.get_dummies(df)
        self.df = df

    def add_historical_windpower(self):
        '''conversion of windspeed into windpower'''
        df = self.df
        t = pd.read_csv('target.csv')
        t['Time'] = pd.to_datetime(t['Time'])
        t.set_index('Time',inplace=True)
        #how does this standardscaler object behave?
        target_scaler = StandardScaler().fit(t)
        t = pd.DataFrame(target_scaler.transform(t),index=t.index,columns=t.columns)
        df = df.join(t,how='left')
        self.target_scaler = target_scaler
        self.df = df

    def add_momentum_force(self):
        '''add momentum'''
        time_lag = 18
        df = self.df
        df['Wind Energy Lag {}'.format(time_lag)] = df['Wind Energy'].shift(time_lag)
        df['Wind Energy Lag {}'.format(2*time_lag)] = df['Wind Energy'].shift(2*time_lag)
        df.dropna(axis=0,inplace=True) ####DROPPING 10 ROWS OF DATA HERE
        # are you not subtracting the future values from present here?
        df['Momentum'] = df['Wind Energy'] - df['Wind Energy Lag {}'.format(time_lag)]
        df['Force'] = df['Wind Energy'] - 2*df['Wind Energy Lag {}'.format(time_lag)] + df['Wind Energy Lag {}'.format(2*time_lag)]
        df.drop(['Wind Energy Lag {}'.format(time_lag),'Wind Energy Lag {}'.format(2*time_lag)],axis=1,inplace=True)
        self.df = df

        ### generate lagged input
        lagged = pd.DataFrame(df['Wind Energy'].shift(1))
        lagged.fillna(method='bfill',inplace=True)
        lagged = StandardScaler().fit_transform(lagged.values)
        self.lagged_input = lagged


    #----GETTER Functions---
    #what are they for?

    def get_df(self):
        return self.df

    def get_lagged_input(self):
        return self.lagged_input

    def get_target_scaler(self):
        return self.target_scaler


'''copy dataframe, transform the data and get the df and lagged input'''
df = df1.copy()
transformer = DataTransformer(df)
transformer.transform()
df = transformer.get_df()
lagged = transformer.get_lagged_input()


'''EDA'''
#you can index both axis without selecting columns?
X = df.loc[:,df.columns!='Wind Energy'].values
y = df['Wind Energy'].values
X = np.concatenate((X,lagged),axis=1)
#how does the concatenate work? is it diff from np.concatenate?


x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.4)
x_train_features = x_train[:,:62]
x_test_features = x_test[:,:62]
x_train_lagged = x_train[:,62]
x_test_lagged = x_test[:,62]
print(x_train_features.shape,x_test_features.shape,x_train_lagged.shape,x_test_lagged.shape)



'''Difference Model'''
