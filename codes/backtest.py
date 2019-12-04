import numpy as np
from filter import Filters
import math
from progress.bar import Bar
import pandas as pd
import cvxpy
import matplotlib.pyplot as plt
import pickle
import os
np.random.seed(1)


class Backtest:
    def __init__(self, path, test_start_loc=0, test_end_loc=None, feeding_win=200, predicting_win=2):
        self.init_cap = 100000
        self.curr_cap = self.init_cap
        self.risk_free = 0.03 / (365 * 48)

        """After having the optimal lambda, backtesting the strategy"""
        optimal_result = pd.read_csv('../data/results/cross_validation_lambda.csv')
        # self.opt_lambda = optimal_result['lambda'][optimal_result['errors'] == optimal_result['errors'].min()].values[0]
        self.opt_lambda = 10
        self.data = pd.read_csv(path, index_col=0, parse_dates=False, names=['time', 'price'])
        self.data['curr_vol'] = (self.data['price'] - self.data['price'].shift(1).fillna(0))
        self.data = self.data.iloc[1:, :]
        self.data['vol'] = (self.data['curr_vol'].rolling(window=feeding_win, min_periods=1).std())
        self.data = self.data.dropna()
        self.test_start_loc = test_start_loc
        self.test_end_loc = test_end_loc if test_end_loc else self.data.shape[0] - predicting_win - feeding_win
        self.feeding_win = feeding_win
        self.predicting_win = predicting_win
        self.cap_history = []
        self.expos = []

    def run(self):
        with Bar('Backtesting', max=self.test_end_loc - self.test_start_loc) as bar:
            for curr_idx in range(self.test_start_loc, self.test_end_loc, self.predicting_win):
                train_data = self.data.iloc[curr_idx:curr_idx + self.feeding_win, 0]
                l1_filter = Filters().filter('l1')
                try:
                    outputs = l1_filter(train_data.values.tolist(), vlambda=self.opt_lambda)
                except cvxpy.error.SolverError:
                    print(f'Failed to Converge of idx {int(curr_idx)}')
                    # pnls.append(None)
                    continue
                trend = outputs[1] - outputs[0]

                for idx in range(self.predicting_win):
                    bar.next()
                    vol = self.data.iloc[curr_idx + self.feeding_win + idx, 2]
                    prev_price = self.data.iloc[curr_idx + self.feeding_win + idx - 1, 0]
                    price = self.data.iloc[curr_idx + self.feeding_win + idx, 0]
                    alpha = trend / (vol)
                    alpha = min(max(alpha, 0), 1)
                    self.curr_cap = self.curr_cap + self.curr_cap * (alpha * (price / prev_price - 1) +
                                                     (1 - alpha) * self.risk_free)
                    self.cap_history.append(self.curr_cap)
                    self.expos.append(alpha)


        with plt.style.context('ggplot'):
            pd.DataFrame(zip(*[self.cap_history, self.expos]), columns=['returns', 'exposure'])\
                .to_csv('../data/results/returns.csv')
            plot_df = pd.DataFrame(zip(*[self.cap_history,
                                         self.data['price'].iloc[self.feeding_win+self.test_start_loc:-self.predicting_win],
                                         self.expos]),
                                   columns=['L1-Filter Returns', 'S&P500', 'exposure'],
                                   index=self.data.index.values[self.feeding_win+self.test_start_loc:-self.predicting_win])
            plot_df['S&P500'] = ((plot_df['S&P500'] / plot_df['S&P500'].shift(1)))
            plot_df['S&P500'] = plot_df['S&P500'].cumprod() * 100000

            fig = plt.figure()

            # Divide the figure into a 2x1 grid, and give me the first section
            ax1 = fig.add_subplot(211)

            # Divide the figure into a 2x1 grid, and give me the second section
            ax2 = fig.add_subplot(212)
            plot_df.plot(y='L1-Filter Returns', ax=ax1)
            plot_df.plot(y='S&P500', ax=ax1)
            plot_df['exposure'].plot.bar(ax=ax2, sharex=True)
            plt.legend()
            plt.title('Backtesting of Strategy')
            labels = [plot_df.index[idx] for idx in range(0, plot_df.shape[0], 100)]
            plt.xticks(range(0, plot_df.shape[0], 100), labels, rotation=45)
            plt.show()


if __name__ == '__main__':
    # bt = Backtest('../data/sp.csv', test_start_loc=0, predicting_win=1, feeding_win=3)
    bt = Backtest('../data/0700.csv', test_start_loc=54000, predicting_win=1, feeding_win=10)
    bt.run()
