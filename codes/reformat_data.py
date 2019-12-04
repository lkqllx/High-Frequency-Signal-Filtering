"""One time program for reformatting original 2330.TW data"""

import pandas as pd
from scipy import fftpack
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import swifter
import datetime as dt
import pandas_datareader as web


def separate_trades_db(file, ticker):
    df = pd.read_csv(file)
    df = df[df['sym'] == '.'.join((ticker, 'HK'))]
    df.to_csv(f'../data/{ticker}.csv')


def reformat_tw_data(file, ticker):
    df = pd.read_csv(file, index_col=0)
    df = df[['lastPx', 'SV1', 'BV1']]
    df = df[df['lastPx'] != 0]
    df.to_csv(f'../data/{ticker}.csv')


def fourier_transform(file):
    """Trial of using fourier transform for changing the signal into frequency domain"""
    df = pd.read_csv(file, index_col=0)
    x = df['lastPx'].values[:1000]
    x = x.tolist()
    sampling = 10
    X = fftpack.fft(x)
    freqs = fftpack.fftfreq(len(x)) * sampling
    fig, ax = plt.subplots()
    ax.stem(freqs, np.abs(X))
    ax.set_xlim(-sampling/2, sampling/2)
    plt.show()


def combine_lambdas(dir_path='../data/results/each_lambda'):
    os.chdir(dir_path)
    file_list = os.listdir('.')
    file_list = [file for file in file_list if file != '.DS_Store']
    file_list = sorted(file_list, key=lambda x: float(re.findall('([\d.]+).csv', x)[0]))
    for file in file_list:
        if re.match('[\d.]+.csv', file):
            lambda_value = re.findall('([\d.]+).csv', file)
            curr_df = pd.read_csv(file, names=lambda_value)[1:]
            try:
                all_df = pd.concat([all_df, curr_df], axis=1, sort=False)
            except:
                all_df = curr_df
    return all_df


def reformat_datetime(files):
    for file in files:
        file_path = os.path.join('../data', file)
        df = pd.read_csv(file_path)
        df.dropna(inplace=True)
        df['Time'] = df.swifter.apply(lambda row: dt.datetime.strptime(' '.join((row['date'], row['time'])),
                                                                       '%Y-%m-%d %H:%M:%S.%f'), axis=1)
        df.index = df['Time'].values
        df.to_csv(os.path.join('../data', re.findall('[a-z]+s([\d.a-z]+)', file)[0]))


def add_volatility(files):
    for file in files:
        file_path = os.path.join('../data', file)
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        df = df.resample('1S').last()
        df = df[(df['cond'] != 'D') & (df['cond'] != 'U')]
        df.dropna(inplace=True)
        df['curr_vol'] = (df['price'] - df['price'].shift(1).fillna(0)) ** 2
        df['vol'] = df['curr_vol'].rolling(window=200).sum() / 200
        df.to_csv(file_path)


def download_price():
    df = web.get_data_yahoo('^GSPC', start='2015-01-01', end='2019-12-02').Close
    df.to_csv('../data/sp.csv')

def resample():
    files = ['../data/0005.csv', '../data/0700.csv']
    for file in files:
        df = pd.read_csv(file, index_col=0, date_parser=pd.to_datetime)
        df = df.resample('5T').last()
        df.dropna(inplace=True)
        df.to_csv(file)


if __name__ == '__main__':
    # separate_trades_db('../data/trades.csv', '0005')
    # separate_trades_db('../data/trades.csv', '0700')
    # reformat_tw_data('../data/2330_ori.csv', '2330')
    # fourier_transform('../data/2330.csv')
    # df = combine_lambdas()

    # reformat_datetime(['trades0005.csv', 'trades0700.csv'])
    # add_volatility(['0005.csv', '0700.csv'])
    # download_price()
    resample()