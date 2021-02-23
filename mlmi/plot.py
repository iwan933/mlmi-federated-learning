from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
from PIL import Image
from torch import Tensor
from torchvision.transforms import functional
from torch.utils import data

from mlmi.participant import BaseTrainingParticipant


def generate_client_label_heatmap(title: str, clients: List['BaseTrainingParticipant'], num_classes: int) -> Tensor:
    dataloaders = [c.train_data_loader for c in clients]
    return generate_data_label_heatmap(title, dataloaders, num_classes)


def generate_data_label_heatmap(title: str, dataloaders: List[data.DataLoader], num_classes: int) -> Tensor:
    labels_list = []
    for dataloader in dataloaders:
        labels = np.array([], dtype=int)
        labels_count = np.zeros((num_classes,))
        for x, y in dataloader:
            labels = np.append(labels, y.numpy())
        labels, counts = np.unique(labels, return_counts=True)
        labels_count[labels] = counts
        labels_list.append(labels_count)
    label_map = np.vstack((*labels_list,))
    # draw heatmap
    fig, ax = plt.subplots(nrows=1, ncols=1)
    sns.heatmap(label_map, ax=ax)
    ax.set_title(title)
    # write to buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    # decode to tensor
    image = Image.open(buf)
    image = functional.to_tensor(image)
    plt.close(fig)
    return image


def generate_confusion_matrix_heatmap(confusion_matrix, title=''):
    import pandas as pd
    import matplotlib.pyplot as plt
    df_cm = pd.DataFrame(confusion_matrix.numpy(), index=[i for i in ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']],
                         columns=[i for i in ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']])
    # draw heatmap
    fig, ax = plt.subplots(nrows=1, ncols=1)
    sns.heatmap(df_cm, ax=ax)
    ax.set_title(title)
    # write to buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    # decode to tensor
    image = Image.open(buf)
    image = functional.to_tensor(image)
    plt.close(fig)
    return image
