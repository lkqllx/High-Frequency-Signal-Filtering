import numpy as np
np.random.seed(1)
from filter import Filters
import math
from progress.bar import Bar
import pandas as pd
import cvxpy
import pickle
import os
from visualization import pye_plots
from reformat_data import combine_lambdas
import pandas_datareader as web
import re
TRAIN_WIN = 500
TEST_WIN = 50


def est_lambda(signal, signal_type: str):

    """Based on the range of lambda to estimate the optimal lambda"""
    if os.path.exists('f../data/results/lambda_{signal_type}.pkl'):
        with open(f'../data/results/lambda_{signal_type}.pkl', 'rb') as f:
            lambda_low, lambda_high = pickle.load(f)
    else:
        lambda_low, lambda_high = est_lambda_range(signal)
        with open(f'../data/results/lambda_{signal_type}.pkl', 'wb') as f:
            pickle.dump([lambda_low, lambda_high], f)

    episodes = 20  # Number of groups for cross validation

    culmulated_outs = []
    lambda_low, lambda_high = lambda_low if lambda_low > 1 else 1, lambda_high
    for episode in range(episodes + 1):
        pnl = 0
        pnls = []
        curr_lambda = round(lambda_low * (lambda_high / lambda_low) ** (episode / episodes), 1)
        with Bar(f'Episode - {episode}', max=math.floor(len(signal) / (TRAIN_WIN + TEST_WIN))) as epoch_bar:
            for group in range(math.floor(len(signal) / (TRAIN_WIN + TEST_WIN))):
                epoch_bar.next()
                train_data = signal[TRAIN_WIN * group:TRAIN_WIN * (group + 1)]
                test_data = signal[TRAIN_WIN * (group + 1):(TRAIN_WIN + TEST_WIN) * (group + 1)]

                l1 = Filters().filter('l1')
                try:
                    trend = l1(train_data, vlambda=curr_lambda)
                except cvxpy.error.SolverError:
                    print(f'Failed to Converge of lambda {int(curr_lambda)}')
                    pnls.append(None)
                    continue
                increment = trend[1] - trend[0]
                pred = predict(last_price=train_data[-1], increment=increment, size=len(test_data))
                profit, ave_profit = cal_error(test_data, pred)
                pnl += ave_profit  # Can be treated as errors
                pnls.append(ave_profit)
        culmulated_outs.append((curr_lambda, np.average(pnls)))
        pd.DataFrame(pnls, columns=['errors']).\
            to_csv(f'../data/results/each_lambda/{curr_lambda}.csv', index=False)
    pd.DataFrame(culmulated_outs, columns=['lambda', 'errors']).\
        to_csv('../data/results/cross_validation_lambda.csv', index=False)

    # print(f'Total Earning - {int(pnl)}\n'
    #       f'Ave Earning - {round(np.average(ave_pnls), 1)}')


def cal_error(test, pred):
    """
    Compute the total pnl for the set.
    Positive result -> algorithm is able to catch the trend and buy the equity at lower price
    Negative result -> algorithm fails to capture trend
    """
    return np.sum(np.square([test_price - pred_price for test_price, pred_price in zip(test, pred)])), \
           np.average(np.square([test_price - pred_price for test_price, pred_price in zip(test, pred)]))


def est_lambda_range(signal):
    """Estimate the upper and lower range of lambda by using different periods of test set"""
    lambda_max = []
    with Bar('Estimate Loading', max=math.floor(len(signal) / (TRAIN_WIN + TEST_WIN))) as est_bar:
        for group in range(math.floor(len(signal) / (TRAIN_WIN + TEST_WIN))):
            est_bar.next()
            test_data = signal[TRAIN_WIN * (group + 1):(TRAIN_WIN + TEST_WIN) * (group + 1)]
            # estimator -> ((D(D)^T)^-1)Dy where D is n-1 by n first order differential matrix
            estimator = Filters().filter('est_lambda_max')
            lambdas = estimator(test_data)
            lambda_max.append(max(lambdas))
    lambda_ave = np.average(lambda_max)
    lambda_std = np.std(lambda_max)
    return lambda_ave - lambda_std * 2, lambda_ave + lambda_std * 2


def predict(last_price, increment, size):
    """Predict the price on the basis of the CONSTANT increment"""
    return [(last_price + idx * increment) for idx in range(size)]


if __name__ == '__main__':
    """Random Fake Signal as Input"""
    # r = RandSignal(upper=10, lower=1, freq=0.01, size=30)
    # signal = noise_signal(r.fake_signal)

    # spx = web.get_data_yahoo('^SPX').Close

    """Real Data as Input"""
    signal = pd.read_csv('../data/0700.csv', index_col=0, parse_dates=True)
    signal = signal[(signal['cond'] != 'D') & (signal['cond'] != 'U')]
    signal.index = pd.to_datetime(signal['time'].values)
    prices = signal['price'].resample('5S').mean()
    prices.dropna(inplace=True)
    est_lambda(prices.values.tolist(), '700hk')

    df = combine_lambdas()
    pye_plots(df, title='Performance of different lambdas',
                   save_to='/Users/andrew/Desktop/HKUST/Courses/DB_filter/figs/lambda_perf.html')
