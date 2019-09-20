"""Place to create filters"""

import cvxpy as cp
import numpy as np
from scipy.sparse import spdiags
from simulation import RandSignal, noise_signal
import matplotlib.pyplot as plt


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
    def l1(data: list, vlambda: int = 10):
        length = len(data)
        e = np.ones((1, length))
        D = spdiags(np.vstack((e, -2*e, e)), range(3), length-2, length)
        x = cp.Variable(shape=length)
        obj = cp.Minimize(0.5 * cp.sum_squares(data - x)
                          + vlambda * cp.norm(D*x, 1))
        prob = cp.Problem(obj)
        prob.solve(solver=cp.CVXOPT, verbose=True)
        print('Solver status: {}'.format(prob.status))
        if prob.status != cp.OPTIMAL:
            raise Exception("Solver did not converge!")
        print("optimal objective value: {}".format(obj.value))
        return x.value

    @staticmethod
    def l2(data: list, vlambda: int = 10):
        length = len(data)
        e = np.ones((1, length))
        D = spdiags(np.vstack((e, -2*e, e)), range(3), length-2, length)
        x = cp.Variable(shape=length)
        obj = cp.Minimize(0.5 * cp.sum_squares(data - x)
                          + vlambda * cp.norm(D*x, 2))
        prob = cp.Problem(obj)
        prob.solve(solver=cp.CVXOPT, verbose=True)
        print('Solver status: {}'.format(prob.status))
        if prob.status != cp.OPTIMAL:
            raise Exception("Solver did not converge!")
        print("optimal objective value: {}".format(obj.value))
        return x.value


if __name__ == '__main__':
    r = RandSignal(upper=10, lower=1, freq=0.1, size=10)
    clean_signal = list(r.fake_signal)
    noise_signal = list(noise_signal(clean_signal))
    l1 = Filters().filter('l1')
    filtered_value = l1(noise_signal)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    font = {'weight': 'normal',
            'size': 16}
    plt.rc('font', **font)
    plt.figure(figsize=(6, 6))
    plt.plot(np.arange(1,len(clean_signal)+1), clean_signal, 'k:', linewidth=1.0)
    plt.plot(np.arange(1,len(clean_signal)+1), np.array(filtered_value), 'b-', linewidth=2.0)
    plt.xlabel('date')
    plt.ylabel('log price')
    plt.show()