from typing import List

import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from presentation.ham10kplots import plot_confusion_matrix, plot_label_counts, plot_performance_graphs

"""
Plots of tensorboard results with adjusted theming for presentation
"""

"""
Label barchart
"""
label_counts = [327, 514, 1099, 115, 1113, 6705, 142]

"""
Confusion matrices
"""
fedavg_personalized_pretrained = np.array([
    [0.81, 0, 0.03, 0.24, 0.02, 0, 0],
    [0, 0.84, 0.01, 0, 0, 0.01, 0.06],
    [0.07, 0.05, 0.83, 0.05, 0.05, 0.04, 0.12],
    [0.02, 0, 0.01, 0.62, 0.01, 0, 0],
    [0.07, 0.02, 0.04, 0.05, 0.75, 0.07, 0],
    [0.02, 0.1, 0.08, 0.05, 0.17, 0.88, 0.21],
    [0, 0, 0, 0, 0.01, 0, 0.61]
])
fedavg_personalized = np.array([
    [0.7, 0.02, 0.06, 0.00, 0.02, 0.01, 0.25],
    [0, 0.73, 0.03, 0, 0, 0.01, 0],
    [0.05, 0.03, 0.68, 0, 0.05, 0.09, 0.12],
    [0.08, 0.11, 0.02, 0.50, 0.01, 0.01, 0],
    [0, 0, 0.05, 0, 0.73, 0.09, 0],
    [0.16, 0.08, 0.15, 0.50, 0.19, 0.77, 0.25],
    [0, 0.03, 0.01, 0, 0, 0.02, 0.38]
])
fedavg_global_pretrained = np.array([
    [0.64, 0.09, 0.06, 0, 0.03, 0, 0],
    [0.02, 0.71, 0.01, 0.13, 0.03, 0.01, 0.09],
    [0.13, 0.1, 0.72, 0, 0.04, 0.06, 0],
    [0.02, 0, 0.02, 0.8, 0.01, 0, 0],
    [0.18, 0.02, 0.12, 0.07, 0.80, 0.12, 0],
    [0, 0.09, 0.07, 0, 0.09, 0.8, 0.05],
    [0, 0, 0.01, 0, 0.01, 0, 0.86]
])
fedavg_global = np.array([
    [0.45, 0.43, 0.08, 1, 0.03, 0.02, 0.06],
    [0.21, 0.14, 0.05, 0, 0.03, 0.02, 0.29],
    [0.16, 0.29, 0.48, 0, 0.18, 0.13, 0.25],
    [0.08, 0, 0.04, 0, 0, 0.01, 0.07],
    [0.08, 0.14, 0.19, 0, 0.64, 0.17, 0.03],
    [0.03, 0, 0.15, 0, 0.13, 0.65, 0.18],
    [0, 0, 0.01, 0, 0, 0.01, 0.12]
])
hierarchical_personalized = np.array([
    [0.68, 0.03, 0.04, 0.4, 0.01, 0.01, 0.1],
    [0, 0.65, 0.03, 0, 0, 0.02, 0],
    [0.1, 0.06, 0.73, 0.2, 0.05, 0.1, 0.05],
    [0.08, 0.1, 0.02, 0.2, 0.01, 0.01, 0],
    [0, 0, 0.03, 0, 0.7, 0.08, 0.05],
    [0.15, 0.12, 0.15, 0.20, 0.22, 0.77, 0.48],
    [0, 0.04, 0, 0, 0.01, 0.02, 0.33]
])
hierarchical_global = np.array([
    [0.64, 0.26, 0.04, 0.18, 0.05, 0.01, 0.04],
    [0, 0.4, 0.04, 0.26, 0.02, 0.01, 0.38],
    [0.09, 0.11, 0.56, 0.36, 0.22, 0.1, 0.08],
    [0.09, 0.08, 0.02, 0.1, 0.02, 0.01, 0],
    [0.18, 0.02, 0.14, 0.03, 0.51, 0.13, 0.04],
    [0, 0.08, 0.19, 0.08, 0.19, 0.71, 0.33],
    [0, 0.05, 0.01, 0, 0, 0.02, 0.12]
])


confusion_matrices = [
    (fedavg_personalized, 'FedAv Personalized'),
    (fedavg_personalized_pretrained, 'FedAv Personalized Pretrained'),
    (hierarchical_personalized, 'FL+HC Personalized'),
    (hierarchical_global, 'Hierarchical Global'),
    (fedavg_global, 'FedAv Global'),
    (fedavg_global_pretrained, 'FedAv Global Pretrained')
]

GREEN = '#95C760'
RED = '#BA4B4D'
BLUE = '#497DE8'
PURPLE = '#B01CF8'
STRIPPED = '--'
NORMAL = '-'
DOTTED = ':'
YELLOW = '#E5AF48'

graph_fileexports = [
    ('FedAvg Pretrained', YELLOW, NORMAL, 'run-ham10k_ham10k_bs16lr1.00E-02cf0.30e1_optSGD_rgb_default_version_0-tag-global_performance_acc_test_mean.csv'),
    ('FedAvg Personalized Pretrained', GREEN, NORMAL, 'run-ham10k_ham10k_bs16lr1.00E-02cf0.30e1_optSGD_rgb_default_version_0-tag-global_performance_personalized_acc_test_mean.csv'),
    ('FL+HC Pretrained', PURPLE, NORMAL, 'run-ham10k_ham10k_bs16lr1.00E-02cf0.30e1_optSGD_rgb_w_dist_eu200.00ri20rc130_default_version_0-tag-global_performance_acc_test_mean.csv'),
    ('FL+HC Personalized Pretrained', PURPLE, NORMAL, 'run-ham10k_ham10k_bs16lr1.00E-02cf0.30e1_optSGD_rgb_w_dist_eu200.00ri20rc130_default_version_0-tag-global_performance_acc_test_mean.csv'),
    ('FedAvg Personalized Pretrained', PURPLE, NORMAL, 'run-ham10k_ham10k_bs16lr1.00E-02cf0.30e1_optSGD_rgb_w_dist_eu200.00ri20rc130_default_version_0-tag-global_performance_personalized_acc_test_mean.csv'),
    ('FedAvg', BLUE, NORMAL, 'run-ham10k_nopretrain_ham10k_bs16lr7.00E-03cf0.30e1_optSGD_rgb_default_version_0-tag-global_performance_acc_test_mean.csv'),
    ('FedAvg Personalized', BLUE, NORMAL, 'run-ham10k_nopretrain_ham10k_bs16lr7.00E-03cf0.30e1_optSGD_rgb_default_version_0-tag-global_performance_personalized_acc_test_mean.csv'),
    ('FL+HC', RED, NORMAL, 'run-ham10k_nopretrain_ham10k_bs16lr7.00E-03cf0.30e1_optSGD_rgb_w_dist_eu300.00ri10rc140_default_version_0-tag-global_performance_acc_test_mean.csv'),
    ('FL+HC Personalized', RED, NORMAL, 'run-ham10k_nopretrain_ham10k_bs16lr7.00E-03cf0.30e1_optSGD_rgb_w_dist_eu300.00ri10rc140_default_version_0-tag-global_performance_personalized_acc_test_mean.csv'),
    ('Reptile Pretrained', GREEN, NORMAL, 'run-ham10kreptile_ham10k_seed123123123_trainc27testc0_sgd_is32ilr0007ib16_ms1000mlri1mlrf0mb5_isev32ect-1ecf-1_default_version_0-tag-train-test_acc_test_mean.csv')
]


if __name__ == '__main__':
    from mlmi.settings import REPO_ROOT
    export_dir = REPO_ROOT / 'data' / 'tensorboard_exports'
    graph_exports_dict = {}
    for title, color, linestyle, filename in graph_fileexports:
        filepath = str((export_dir / filename).absolute())
        df = pd.read_csv(filepath, sep=',', index_col='Step')
        # apply exponential smoothing
        df['Value'] = df['Value'].ewm(alpha=0.2).mean()
        graph_exports_dict[title] = (title, color, linestyle, df['Value'])


    plot_performance_graphs([graph_exports_dict[key] for key in
                             ['FedAvg Pretrained', 'FL+HC Pretrained', 'FedAvg', 'FL+HC']])


    #plot_label_counts(label_counts)

    #for confusion_matrix, title in confusion_matrices:
    #    plot_confusion_matrix(confusion_matrix, title)

    plt.show()
