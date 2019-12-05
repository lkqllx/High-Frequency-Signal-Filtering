"""Place to create filters"""
import numpy as np
import cvxpy as cp
import cvxopt
from scipy.sparse import spdiags
from simulation import RandSignal, noise_signal
from visualization import plot_two_lines
import pandas as pd
from utility import dif1_matrix
import pandas_datareader as web
np.random.seed(1)

class Filters:
    """
    Class Filters
    It is the class for creating varying types of filters

    SAMPLE:
        ->  l1 filter: l1 = Filters().filter('l1')
        ->  l2 filter: l2 = Filters().filter('l2')
    """
    def filter(self, type):
        return eval(f'self.{type}')

    @staticmethod
    def l1(data: list, vlambda: int = 10, verbose=False):
        length = len(data)
        e = np.ones((1, length))
        D = spdiags(np.vstack((e, -2*e, e)), range(3), length-2, length)
        x = cp.Variable(shape=length)
        obj = cp.Minimize(0.5 * cp.sum_squares(data - x)
                          + vlambda * cp.norm(D*x, 1))
        prob = cp.Problem(obj)
        prob.solve(solver=cp.CVXOPT, verbose=verbose)
        # print('Solver status: {}'.format(prob.status))
        if prob.status != cp.OPTIMAL:
            raise Exception("Solver did not converge!")
        # print("optimal objective value: {}".format(obj.value))
        return x.value

    @staticmethod
    def l2(data: list, vlambda: int = 10, verbose=False):
        length = len(data)
        e = np.ones((1, length))
        D = spdiags(np.vstack((e, -2*e, e)), range(3), length-2, length)
        x = cp.Variable(shape=length)
        obj = cp.Minimize(0.5 * cp.sum_squares(data - x)
                          + vlambda * cp.norm(D*x, 2))
        prob = cp.Problem(obj)
        prob.solve(solver=cp.CVXOPT, verbose=verbose)
        print('Solver status: {}'.format(prob.status))
        if prob.status != cp.OPTIMAL:
            raise Exception("Solver did not converge!")
        print("optimal objective value: {}".format(obj.value))
        return x.value

    @staticmethod
    def est_lambda_max(y):
        length = len(y)
        D = dif1_matrix(length)[1:, :]
        lambda_max =  np.matmul(np.matmul(np.linalg.inv(np.matmul(D, np.transpose(D))), D), y)
        return lambda_max


if __name__ == '__main__':
    df = pd.read_csv('../data/0005.csv', index_col=0, names=['date', 'price'])
    df = df.iloc[:int(0.1 * df.shape[0]), 0]
    # df = df.price.rolling(window=5, win_type='hamming').mean()
    # df.dropna(inplace=True)

    # df = web.get_data_yahoo('^GSPC').Close[700:]
    # df.to_csv('../data/sp.csv')
    # dates = df.index.values
    # df = pd.Series(np.log(df), index=dates)

    # r = RandSignal(upper=10, lower=1, freq=0.5, size=10)
    # clean_signal = list(r.fake_signal)
    # df = list(noise_signal(clean_signal))
    l1 = Filters().filter('l1')
    filtered_value = l1(df.values.tolist(), vlambda=8)

    plot_two_lines(df, pd.Series(filtered_value, index=df.index), to_png=False, filename='filtered_fake.png')