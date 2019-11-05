"""One time program for reformatting original 2330.TW data"""

import pandas as pd
from scipy import fftpack
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import swifter
import datetime as dt

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
        df.to_csv(file_path)



if __name__ == '__main__':
    # separate_trades_db('../data/trades.csv', '0005')
    # separate_trades_db('../data/trades.csv', '0700')
    # reformat_tw_data('../data/2330_ori.csv', '2330')
    # fourier_transform('../data/2330.csv')
    # df = combine_lambdas()

    reformat_datetime(['0005.csv', '0700.csv'])