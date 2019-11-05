import numpy as np
from filter import Filters
import math
from progress.bar import Bar
import pandas as pd
import cvxpy
import pickle
import os
np.random.seed(1)


class Backtest:
    def __init__(self, path, test_start_loc=15000, test_end_loc=20000, feeding_win=600, predicting_win=10):
        self.init_cap = 100000
        self.curr_cap = self.init_cap
        self.risk_free = 0.02

        """After having the optimal lambda, backtesting the strategy"""
        optimal_result = pd.read_csv('../data/results/cross_validation_lambda.csv')
        self.opt_lambda = optimal_result['lambda'][optimal_result['errors'] == optimal_result['errors'].min()].values[0]

        self.data = pd.read_csv(path, index_col=0, parse_dates=True)
        self.test_start_loc = test_start_loc
        self.test_end_loc = test_end_loc
        self.feeding_win = feeding_win
        self.predicting_win = predicting_win
        self.cap_history = []

    def run(self):
        with Bar('Backtesting', max=self.test_end_loc - self.test_start_loc) as bar:
            for curr_idx in range(self.test_start_loc, self.test_end_loc, self.predicting_win):
                train_data = self.data.iloc[curr_idx:curr_idx + self.feeding_win, 3]
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
                    vol = self.data.iloc[curr_idx + idx, 8]
                    prev_price = self.data.iloc[curr_idx + idx - 1, 3]
                    price = self.data.iloc[curr_idx + idx, 3]
                    alpha = trend / (np.sqrt(vol))
                    alpha = min(max(alpha, -1), 1)
                    self.curr_cap = self.curr_cap + (alpha * (price / prev_price - 1) +
                                                     (1 - alpha) * self.risk_free)

                    self.cap_history.append(self.curr_cap)

        pd.DataFrame(self.cap_history, columns=['returns']).to_csv('data/results/returns.csv')




if __name__ == '__main__':
    bt = Backtest('../data/0700.csv')
    bt.run()