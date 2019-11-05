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
    def __init__(self, path, start_loc=15000):
        self.init_wealth = 100000

        """After having the optimal lambda, backtesting the strategy"""
        optimal_result = pd.read_csv('../data/cross_validation_lambda.csv')
        self.opt_lambda = optimal_result['lambda'][optimal_result['error'] == optimal_result['error'].min()]

        self.data = pd.read_csv(path, index_col=0, parse_dates=True)[start_loc:]



    def run():



if __name__ == '__main__':
    backtest()