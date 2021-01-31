from mlmi.structs import FederatedDatasetData


def plot_client_label_heatmap(fed_dataset: 'FederatedDatasetData'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    labels_list = []
    for k, dataloader in fed_dataset.train_data_local_dict.items():
        labels = np.array([], dtype=int)
        labels_count = np.zeros((fed_dataset.class_num,))
        for x, y in dataloader:
            labels = np.append(labels, y.numpy())
        labels, counts = np.unique(labels, return_counts=True)
        labels_count[labels] = counts
        labels_list.append(labels_count)
    label_map = np.vstack((*labels_list,))
    sns.heatmap(label_map)
    plt.show()
