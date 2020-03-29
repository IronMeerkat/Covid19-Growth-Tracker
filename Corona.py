from math import sqrt, log
from random import random as r

import pickle

import numpy as np
import matplotlib
from  matplotlib import pyplot as plt
import pandas as pd
from matplotlib import rcParams
from scipy.optimize import curve_fit

from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error as mse, median_absolute_error as mae
from sklearn.preprocessing import StandardScaler

# import tensorflow as tf
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import Adam, SGD

rcParams['figure.figsize'] = (10, 6)
rcParams['legend.fontsize'] = 16
rcParams['axes.labelsize'] = 16

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

Expo = pickle.load(open("ExpoModel.p", "rb"))

def longest(l):
    max = 0
    for i in l:
        if max < len(i):
            max = len(i)
    return max


def ex(x, w=0, b=0):
    return w * b ** x

def mean_squared_error(y_true, y_pred):
    s = (y_true - y_pred)**2
    return s.mean()

def Land(country, province=None, start_date=None, end_date=None, casetype=['confirmed', 'death', 'recovered'],
                      cumsum=True):
    df = pd.read_csv('https://raw.githubusercontent.com/RamiKrispin/coronavirus-csv/master/coronavirus_dataset.csv',
                     names=["province_state", 'country_region', 'lat', 'long', 'date', 'cases', 'type'], skiprows=1,
                     parse_dates=['date'])

    df = df[df.type.isin(casetype)]

    if start_date is not None:
        df = df[df.date >= start_date]
    else:
        if end_date is not None:
            df = df[df.date <= end_date]

    df = df[(df.country_region == country)]

    if province is not None:
        df = df[(df.province_state == province)]


    df = df.pivot_table(index=["date", 'country_region'], columns='type', values='cases', \
                        aggfunc={'date': 'first', 'country_region': 'first', 'cases': np.sum}) \
        .reset_index().fillna(0)

    if 'death' not in df.columns:
        df['death'] = 0

    if 'recovered' not in df.columns:
        df['recovered'] = 0

    df.sort_values(by=['country_region', 'date'], inplace=True)


    if cumsum is True:
        df.confirmed = df.groupby('country_region')['confirmed'].transform(pd.Series.cumsum)
        df.recovered = df.groupby('country_region')['recovered'].transform(pd.Series.cumsum)
        df.death = df.groupby('country_region')['death'].transform(pd.Series.cumsum)


    df = df.reset_index()
    df = df[df['confirmed'] > 0]
    len = df.shape[0]
    df.insert(1, 'day', [i for i in range(1, len + 1)], True)
    df = df.set_index("day", drop = False)
    return df

def doubleTime(df):
    sum = 0.0
    arr = df['confirmed']
    for i in range(df.shape[0]-1):
        sum += arr.iat[i+1]/arr.iat[i]
    try:
        return log(2,float(sum/(df.shape[0]-1)))
    except:
        return 99.9999

def doubleRate(lands):
    doubling = {}
    pulse = 7

    for k, v in lands.items():
        doubling[k] = []
        f = [j for j in range(v.shape[0]) if j % pulse == 0]
        temp = v.tail(f[-1])
        for i in f[:-1]:
            doubling[k].append(doubleTime(temp.iloc[i:i + pulse - 1]))

    l = longest(list(doubling.values()))

    for k, v in doubling.items():
        delt = l - len(v)
        for i in range(delt):
            v.insert(0, 0.0)

    doubleMatrix = pd.DataFrame(doubling)
    doubleMatrix.index.name = "week"
    doubleMatrix.rename_axis = "Average time to double cases (in days)"
    return doubleMatrix

def manyLands(la):
    Lands = {}
    for i in la:
        Lands[i] = Land(i)
    return Lands

def curve(lands, value='confirmed'):
    for k, v in lands.items():
        p = np.polyfit(x=v['day'], y=v[value], deg=3)
        e = np.polyfit(x=v['day'], y=np.log(v[value]), deg=1)
        v.insert(2, '%s predicted_by_polynomial'%value,
                 [p[0] * i ** 3 + p[1] * i ** 2 + p[2] * i + p[3] for i in range(1, v.shape[0] + 1)],
                 True)
        v.insert(3, '%s predicted_by_exponential'%value, [np.e ** (e[0] * i + e[1]) for i in range(1, v.shape[0] + 1)], True)

    ct = {}
    for k, v in lands.items():
        P = 0
        E = 0
        for i in range(v.shape[0]):
            P = P + (v['predicted_by_polynomial'].iat[i] - v['confirmed'].iat[i]) ** 2
            E = E + (v['predicted_by_exponential'].iat[i] - v['confirmed'].iat[i]) ** 2
        ct[k] = [P, E, sqrt(E / P)]

    curveTest = pd.DataFrame(ct)
    curveTest.index = ['Polynomial^2 ', 'Exponential^2', 'Ratio']
    return curveTest

def rawData(lands):
    data = {}
    l = max([i.shape[0] for i in lands.values()])
    for k, v in lands.items():
        data[k] =[0 for i in range(l-v.shape[0])] + list(v['confirmed'])
    df = pd.DataFrame(data)
    df.insert(0, 'day', [i for i in range(1, df.shape[0] + 1)], True)
    df.set_index("day", drop=False)
    return df

def excludeChina(cumsum=True):
    df = pd.read_csv('https://raw.githubusercontent.com/RamiKrispin/coronavirus-csv/master/coronavirus_dataset.csv',
                     names=["province_state", 'country_region', 'lat', 'long', 'date', 'cases', 'type'], skiprows=1,
                     parse_dates=['date'])

    df = df[(df.country_region != 'China')]


    df = pd.pivot_table(df, columns=['type'], index=['date'], values=['cases'], aggfunc=np.sum)

    df = df['cases']
    df = df.reset_index()
    len = df.shape[0]
    df.insert(1, 'day', [i for i in range(1, len + 1)], True)


    if cumsum:
        df['confirmed'] = df['confirmed'].cumsum()
        #df['recovered'] = df['recovered'].cumsum()
        df['death'] = df['death'].cumsum()

    return df

def predict(df,days, fromToday = True, curve='exponential_fit', value='confirmed'):
    size = df.shape[0]
    predictTo = days
    if fromToday:
       predictTo += size

    if curve == 'exponential_fit':
        X = np.array([i for i in range(1, predictTo+1)]).reshape(-1, 1)
        y = np.e**(Expo.predict(X))
        return y.astype(int)

    if curve == 'polynomial':
        p = np.polyfit(x=df['day'], y=df[value], deg=3)
        x = predictTo
        y = int(p[0]*x**2+p[1*x+p[3]])
        return y




noChina = excludeChina()
for k, v in noChina.items():
    print(k)

exit()
lands = manyLands(['Israel', 'Italy', 'US', 'Sweden', 'Germany'])



for k, v in lands.items():
    y = predict(v, days=0).flatten()
    true = np.array(v['confirmed']).flatten()
    this_mae = np.mean(true - y).astype(int)
    print("{}'s metric: {}".format(k,this_mae))