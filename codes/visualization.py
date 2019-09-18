"""
visualization.py contains all the functions for plotting.
"""
import matplotlib.pyplot as plt
import pandas as pd
import json

class VisualTool:
    """
    Class VisualTool for visualizing signal from real dataset or fake dataset
    :param target -> str, list or other iterable
        if type(target) == str -> read csv for plotting
        if type(target) == list -> directly plot list
        if type(target) == other iterables -> list(target) and then plot list

    SAMPLE
    read csv file:
        -> v = VisualTool('../data/0005.csv')
        -> v.plot_line('price' , 0.005) where 'price' is col_name of last price data in csv file

    read list file:
        -> v = VisualTool(noise_signal)
        -> v.plot_line(percent=1)

    """
    def __init__(self, target):
        if isinstance(target, str):
            self.data = pd.read_csv(target, index_col=0)
            self.data.index = pd.to_datetime(self.data.index)
            self._type = 'from_csv'
        elif isinstance(target, list):
            self.data = target
            self._type = 'from_list'
        else:
            self.data = list(target)
            self._type = 'from_list'

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

        if self._type == 'from_list':
            plt.figure(figsize=(20,12))
            plt.plot(range(int(percent*len(self.data))), self.data[:int(percent*len(self.data))])
            plt.show()

if __name__ == '__main__':
    v = VisualTool('../data/0005.csv')
    v.plot_line('price' , 0.005)