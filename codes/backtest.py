from simulation import RandSignal, noise_signal
from filter import Filters
import math
import numpy as np
from progress.bar import Bar
from visualization import VisualTool
import pandas as pd
import cvxpy
np.random.seed(1)


TRAIN_WIN = 500
TEST_WIN = 100


def backtest(signal):

    pnl = 0
    ave_pnls = []
    with Bar('Processing', max=math.floor(len(signal) / (TRAIN_WIN + TEST_WIN))) as bar:
        for group in range(math.floor(len(signal) / (TRAIN_WIN + TEST_WIN))):
            bar.next()
            train_data = signal[TRAIN_WIN * group:TRAIN_WIN * (group + 1)]
            test_data = signal[TRAIN_WIN * (group + 1):(TRAIN_WIN + TEST_WIN) * (group + 1)]
            l1 = Filters().filter('l1')
            try:
                trend = l1(train_data, vlambda=)
            except cvxpy.error.SolverError:
                print('Failed to Converge')
                continue
            increment = trend[1] - trend[0]
            pred = predict(last_price=train_data[-1], increment=increment, size=len(test_data))
            profit, ave_profit = cal_pnl(test_data, pred)
            pnl += profit
            ave_pnls.append(ave_profit)
    print(f'Total Earning - {int(pnl)}\n'
          f'Ave Earning - {round(np.average(ave_pnls), 1)}')


def cal_pnl(test, pred):
    """
    Compute the total pnl for the set.
    Positive result -> algorithm is able to catch the trend and buy the equity at lower price
    Negative result -> algorithm fails to capture trend
    """
    return np.sum([test_price - pred_price for test_price, pred_price in zip(test, pred)]), \
           np.average([test_price - pred_price for test_price, pred_price in zip(test, pred)])


def predict(last_price, increment, size):
    """Predict the price on the basis of the CONSTANT increment"""
    return [(last_price + idx * increment) for idx in range(size)]


if __name__ == '__main__':
    """Random Fake Signal as Input"""
    # r = RandSignal(upper=10, lower=1, freq=0.01, size=30)
    # signal = noise_signal(r.fake_signal)

    """Real Data as Input"""
    signal = pd.read_csv('../data/0700.csv', index_col=0)
    signal = signal[(signal['cond'] != 'D') & (signal['cond'] != 'U')].price.values.tolist()


    # v = VisualTool(signal)
    # v.plot_line(percent=1, to_png=False)
    backtest(list(signal))
