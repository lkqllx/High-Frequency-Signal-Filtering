"""
simulation.py is to manually create an environment for testing the performance of filtering
"""
import numpy as np
from visualization import VisualTool
import random
np.random.seed(1)


def noise_signal(signals, high=1, low=-1):
    """
    Add Noise to signals
    :param signals: clean signals
    :param high: upper limit of noise
    :param low: lower limit of noise
    :return: a noised signal
    """
    for signal in signals:
        yield signal + random.uniform(high, low)


class RandSignal:
    """
    Class RandSignal for generating signals - step or simulated signal
    :param upper: upper limit of random signal
    :param lower: lower limit of random signal
    :param size: the number of periods
    :param freq: the number of padding points per period

    SAMPLE:
    step signal - step-wise signal:
        -> r = RandSignal(upper=10, lower=1, freq=0.1, size=10)
        -> step_signal = list(r.step_signal)
        -> v = VisualTool(step_signal)
        -> v.plot_line(percent=1)

    simulated signal - continuously changing signal:
        -> r = RandSignal(upper=10, lower=1, freq=0.1, size=10)
        -> simulated_signal = list(r.fake_signal)
        -> v = VisualTool(simulated_signal)
        -> v.plot_line(percent=1)
    """
    def __init__(self, upper: int, lower: int, size: int = 10, freq: float = 0.1):
        self.upper = upper
        self.lower = lower
        self._size = size
        self._freq = freq
        self.rand_width = list(self.random_nums)
        self.rand_length = list(self.random_nums)

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

    @property
    def fake_signal(self):
        """
        Create a simulated signal with random length and width
        :return: a list of simulated signal
        """
        result = []
        for idx, length in enumerate(self.rand_length):
            if idx == 0:
                """Initialize the point at (x=0, y=0)"""
                result.append(0)

            y_start = result[-1]
            y_distance = self.rand_width[idx] - result[-1]
            for count in range(int(length / self._freq)):
                # yield (self.rand_width[idx] - start_point) * count / int(length / self._freq)
                result.append(y_start + y_distance * count / int(length / self._freq))
        return result





if __name__ == '__main__':
    r = RandSignal(upper=10, lower=1, freq=0.1, size=10)
    step_signal = list(r.fake_signal)
    v = VisualTool(step_signal)
    v.plot_line(percent=1, to_png=True, png_path='../figs/simulated_clean.png')

    noise_signal = noise_signal(step_signal)
    v = VisualTool(noise_signal)
    v.plot_line(percent=1, to_png=True, png_path='../figs/simulated_noisy.png')





