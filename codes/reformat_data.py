"""One time program for reformatting original 2330.TW data"""

import pandas as pd
from scipy import fftpack
import matplotlib.pyplot as plt
import numpy as np


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


if __name__ == '__main__':
    # separate_trades_db('../data/trades.csv', '0005')
    # separate_trades_db('../data/trades.csv', '0700')
    # reformat_tw_data('../data/2330_ori.csv', '2330')
    fourier_transform('../data/2330.csv')