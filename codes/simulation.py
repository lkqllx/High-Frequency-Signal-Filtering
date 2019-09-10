"""
simulation.py is to manually create an environment for testing the performance of filtering
"""
import numpy as np
from visualization import VisualTool
import random
np.random.seed(1)

class RandSignal:
    def __init__(self, upper: int, lower: int, size: int = 10, freq: float = 0.1):
        self.upper = upper
        self.lower = lower
        self._size = size
        self._freq = freq
        self.rand_width = list(self.random_nums)
        self.rand_length = self.random_nums

    @property
    def random_nums(self):
        """
        Create random numbers
        :return: a generator contains a list of random number
        """
        for _ in range(self._size):
            yield np.random.randint(low=self.lower, high=self.upper)

    @property
    def step_signal(self):
        """
        Create step signals with random width and length
        :return: a generator of step signal
        """
        for idx, length in enumerate(self.rand_length):
            for _ in range(int(length / self._freq)):
                yield self.rand_width[idx]

    def noise_signal(self, signals, high=1, low=-1):
        """
        Add Noise to signals
        :param signals: clean signals
        :param high: upper limit of noise
        :param low: lower limit of noise
        :return: a noised signal
        """
        for signal in signals:
            yield signal + random.uniform(high, low)

if __name__ == '__main__':
    r = RandSignal(upper=5, lower=1, freq=0.05, size=20)
    step_signal = list(r.step_signal)
    v = VisualTool(step_signal)
    v.plot_line(percent=1)

    noise_signal = r.noise_signal(step_signal)
    v = VisualTool(noise_signal)
    v.plot_line(percent=1)





