"""
visualization.py contains all the functions for plotting.
"""
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np
from pyecharts.charts import *
import pyecharts.options as opts
import os
import re

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
        -> v.plot_line(percent=1) # percent means the percentage of points to be plotted

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

    def plot_line(self, col_name: str = None, percent: float = 0.1, to_png: bool = False, png_path: str = None):
        if self._type == 'from_csv':
            self.check_col_name(col_name)
            plt.figure(figsize=(20,12))
            plt.plot(range(int(percent*self.data.shape[0])), self.data[col_name][:int(percent*self.data.shape[0])])
            if to_png:
                plt.savefig(png_path, dpi=300)
                return
            plt.show()

        if self._type == 'from_list':
            plt.figure(figsize=(20,12))
            plt.plot(range(int(percent*len(self.data))), self.data[:int(percent*len(self.data))])
            if to_png:
                plt.savefig(png_path, dpi=300)
                return
            plt.show()


def plot_two_lines(original_signal: pd.Series, filtered_value: pd.Series, to_png=False, filename=None):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    font = {'weight': 'normal',
            'size': 16}
    plt.rc('font', **font)
    fig = plt.figure(figsize=(12, 8), dpi=900)
    # original_signal.plot('b-', linewidth=2.0)
    # filtered_value.plot('r--', linewidth=2.0)
    plt.plot(original_signal.index.values, original_signal.values, color='turquoise', linestyle='solid', linewidth=2.0,
             label='S&P500')
    plt.plot(filtered_value.index.values, filtered_value.values, color='tomato', linestyle='dashed', linewidth=2.0,
             label='L1-Filter Estimated Trend')
    plt.xlabel('date')
    plt.ylabel('log(price)')
    plt.legend()
    if not to_png:
        plt.show()
    else:
        plt.savefig(f'../figs/{filename}')


def plot_pye_lines(data: pd.DataFrame, title):
    data = data.astype(float)
    data = data.round(0)
    data = data.iloc[:, [idx for idx in list(range(0, data.shape[1], 5))]]
    line = Line(init_opts=opts.InitOpts(width="1200px", height="800px"))
    line.add_xaxis(xaxis_data=data.index.values.tolist())
    line.set_global_opts(
            datazoom_opts=opts.DataZoomOpts(),
            legend_opts=opts.LegendOpts(pos_top="0%", pos_right='0%', pos_left='90%'),
            title_opts=opts.TitleOpts(title=title.upper(), pos_left='0%'),
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross", is_show=True),
            xaxis_opts=opts.AxisOpts(boundary_gap=False, max_interval=5),
            yaxis_opts=opts.AxisOpts(
                axislabel_opts=opts.LabelOpts(formatter="{value}"),
                splitline_opts=opts.SplitLineOpts(is_show=True),
            )
        )
    line.set_series_opts(
            markpoint_opts=opts.MarkPointOpts(data=[opts.MarkPointItem(type_='max', name='Max'),
                                                    opts.MarkPointItem(type_='min', name='Min')]),
        )
    for idx, column in enumerate(data.columns):
        line.add_yaxis(y_axis=data.iloc[:, idx].values.tolist(),
                       series_name=column,
                       is_smooth=True,
                       label_opts=opts.LabelOpts(is_show=False),
                       linestyle_opts=opts.LineStyleOpts(width=2)
                       )

    return line


def plot_profit_line(path, title):
    if isinstance(path, str):
        df = pd.read_csv(path)
        df['errors'] = df['errors'].astype(int)
    else:
        df = path
    line = Line(init_opts=opts.InitOpts(width="1200px", height="600px"))
    line.add_xaxis(xaxis_data=list(map(str, df.iloc[:, 0].values.tolist())))  # input of x-axis has been string format
    line.add_yaxis(y_axis=df.iloc[:, 1].values.tolist(),
                   series_name=title.title(),
                   is_smooth=True,
                   label_opts=opts.LabelOpts(is_show=False),
                   linestyle_opts=opts.LineStyleOpts(width=3)
                   )
    line.set_global_opts(
            datazoom_opts=opts.DataZoomOpts(),
            legend_opts=opts.LegendOpts(pos_top="20%", pos_right='0%', pos_left='90%'),
            title_opts=opts.TitleOpts(title=title.upper(), pos_left='0%'),
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross", is_show=True),
            xaxis_opts=opts.AxisOpts(boundary_gap=False, max_interval=5),
            yaxis_opts=opts.AxisOpts(
                axislabel_opts=opts.LabelOpts(formatter="{value}"),
                splitline_opts=opts.SplitLineOpts(is_show=True),
            )
        )
    line.set_series_opts(
            markpoint_opts=opts.MarkPointOpts(data=[opts.MarkPointItem(type_='max', name='Max'),
                                                    opts.MarkPointItem(type_='min', name='Min')]),
        )

    return line


def compute_risk_adjusted_return(dir_path='/Users/andrew/Desktop/HKUST/Courses/DB_filter/data/results/each_lambda'):
    os.chdir(dir_path)
    file_list = os.listdir('.')
    file_list = [file for file in file_list if file != '.DS_Store']
    file_list = sorted(file_list, key=lambda x: float(re.findall('([\d.]+).csv', x)[0]))
    for file in file_list:
        if re.match('[\d.]+.csv', file):
            lambda_value = re.findall('([\d.]+).csv', file)
            curr_df = pd.read_csv(file, names=lambda_value)[1:]
            try:
                all_df = pd.concat([all_df, curr_df], axis=1, sort=False)
            except:
                all_df = curr_df
    all_df = all_df.astype(float)
    # all_df.dropna(inplace=True)
    sharpe_list = [(vlambda, round(all_df[vlambda].mean() / all_df[vlambda].std(), 2)) for vlambda in all_df.columns]
    return pd.DataFrame(sharpe_list, columns=['lambda', 'Risk-adjusted Return'])


def pye_plots(data: pd.DataFrame, title, save_to):

    line_plot = plot_pye_lines(data, title)
    profit_plot = plot_profit_line('~/Desktop/HKUST/Courses/DB_filter/data/results/cross_validation_lambda.csv',
                                   'Total Profit')

    risk_adjusted_plot = plot_profit_line(compute_risk_adjusted_return(), 'Risk-Adjusted Return')
    Page().add(*[line_plot, profit_plot, risk_adjusted_plot]).render(path=save_to)


if __name__ == '__main__':
    v = VisualTool('../data/0005.csv')
    v.plot_line('price', 0.1, True, '../figs/0005_HK_Plot.png')