"""Place to create filters"""

import cvxpy as cp
import numpy as np
from scipy.sparse import spdiags
from simulation import RandSignal, noise_signal
from visualization import plot_two_lines
import pandas as pd
np.random.seed(1)

class Filters:
    """
    Class Filters
    It is the class for creating varying types of filters

    SAMPLE:
        ->  l1 filter: l1 = Filters().filter('l1')
        ->  l2 filter: l2 = Filters().filter('l2')
    """
    def filter(self, _type):
        return eval(f'self.{_type}')

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


if __name__ == '__main__':
    df = pd.read_csv('../data/0005.csv')[:400]
    # df = df.price.rolling(window=5, win_type='hamming').mean()
    # df.dropna(inplace=True)
    r = RandSignal(upper=10, lower=1, freq=0.1, size=10)
    clean_signal = list(r.fake_signal)
    noise_signal = list(noise_signal(clean_signal))
    l1 = Filters().filter('l1')
    filtered_value = l1(df.price.values.tolist(), vlambda=50)

    plot_two_lines(df.price.values.tolist(), filtered_value, to_png=False, filename='filtered_fake.png')
