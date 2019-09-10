"""
visualization.py contains all the functions for plotting.
"""

import matplotlib.pyplot as plt
import pandas as pd
import json

class VisualTool:

    def __init__(self, target):
        if isinstance(target, str):
            self.data = pd.read_csv(target, index_col=0)
            self.data.index = pd.to_datetime(self.data.index)
            self._type = 'from_csv'
        elif isinstance(target, list):
            self.data = target
            self._type = 'from_data'
        else:
            self.data = list(target)
            self._type = 'from_data'

    def check_col_name(self, col_name):
        if col_name not in self.data.columns:
            print('There is no such column - {}'.format(col_name))
            available_names = json.dumps(self.data.columns.values.tolist(), indent=2)
            print('The available names:')
            print(available_names)
            raise KeyError

    def plot_line(self, col_name: str = None, percent: float = 0.1):

        if self._type == 'from_csv':
            self.check_col_name(col_name)
            plt.figure(figsize=(20,12))
            plt.plot(range(int(percent*self.data.shape[0])), self.data[col_name][:int(percent*self.data.shape[0])])
            plt.show()

        if self._type == 'from_data':
            plt.figure(figsize=(20,12))
            plt.plot(range(int(percent*len(self.data))), self.data[:int(percent*len(self.data))])
            plt.show()


if __name__ == '__main__':
    v = VisualTool('../data/0005.csv')
    v.plot_line('lastPx' , 0.005)


