from typing import List, Tuple

import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

"""
Plots of tensorboard results with adjusted theming for presentation
"""
label_dict = {0: 'akiec', 1: 'bcc', 2: 'bkl', 3: 'df', 4: 'mel', 5: 'nv', 6: 'vasc'}
sns.set_context(rc={'patch.linewidth': 0.0})

bg_color = '#DAEDEF'
first_color = '#ADC9C4'
second_color = '#7D918E'


def set_plot_theme(ax):
    ax.set_facecolor(bg_color)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_color(second_color)
    ax.xaxis.label.set_color(second_color)
    ax.yaxis.label.set_color(second_color)
    ax.yaxis.grid(color=second_color, linewidth=.5, zorder=0)
    ax.tick_params(axis='x', colors=second_color)
    ax.tick_params(axis='y', colors=second_color, width=.5)


def plot_label_counts(label_counts):
    series = pd.Series(label_counts, index=[label_dict[i] for i in range(7)])
    fig, ax = plt.subplots(nrows=1, ncols=1, facecolor=bg_color)
    ax.set_title('', color=second_color)
    sns.barplot(x=series.index, y=series, ax=ax, ci=None, color=first_color, zorder=3)
    set_plot_theme(ax)
    fig.show()


def plot_confusion_matrix(confusion_matrix, title):
    pct_matrix = confusion_matrix / np.sum(confusion_matrix, axis=0)
    df_cm = pd.DataFrame(pct_matrix,
                         index=[label_dict[i] for i in range(7)],
                         columns=[label_dict[i] for i in range(7)])
    # draw heatmap
    fig, ax = plt.subplots(nrows=1, ncols=1, facecolor=bg_color)
    cmap = sns.dark_palette("#E3F8FA", as_cmap=True)
    sns.heatmap(df_cm, ax=ax, annot=True, fmt=".2f", cmap=cmap)
    ax.set_title(title, color=second_color)
    ax.spines['left'].set_color(second_color)
    ax.spines['left'].set_visible(True)
    ax.spines['right'].set_color(second_color)
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_color(second_color)
    ax.spines['top'].set_visible(True)
    ax.spines['bottom'].set_color(second_color)
    ax.spines['bottom'].set_visible(True)
    ax.xaxis.label.set_color(second_color)
    ax.yaxis.label.set_color(second_color)
    ax.tick_params(axis='x', colors=second_color, width=1.0)
    ax.tick_params(axis='y', colors=second_color, width=.5)
    fig.show()


def plot_performance_graphs(data: List[Tuple[str, str, str, pd.Series]]):
    fig, ax = plt.subplots(nrows=1, ncols=1, facecolor=bg_color)
    ax.set_ylim([0.0, 1.0])
    set_plot_theme(ax)
    for title, color, linestyle, series in data:
        ax.plot(series.index, series, label=title, color=color, linestyle=linestyle)
        #plt.axvline(x=8, color=second_color)
    ax.legend()
    fig.show()
