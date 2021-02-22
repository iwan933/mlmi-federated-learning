import numpy as np
from typing import Dict, Optional

import torch
from torch import Tensor

from mlmi.log import getLogger
from mlmi.participant import BaseTrainingParticipant


logger = getLogger(__name__)


def estimate_weights(labels):
    weight = torch.ones((7,))
    label, counts = torch.unique(labels, return_counts=True)
    weight[label] = weight[label] - counts / torch.sum(counts)
    return weight


class FedAvgHam10kClient(BaseTrainingParticipant):

    def get_model_kwargs(self) -> Optional[Dict]:
        labels = torch.LongTensor([])
        for _, y in self.train_data_loader:
            labels = torch.cat((labels, y))
        for _, y in self.test_data_loader:
            labels = torch.cat((labels, y))
        weights = estimate_weights(labels)
        return {
            'weights': weights
        }


def initialize_ham10k_clients(context: 'FedAvgExperimentContext', dataset: 'FederatedDatasetData',
                              initial_model_state: Dict[str, Tensor]):
    clients = []
    logger.debug('... creating total of {} clients'.format(len(dataset.train_data_local_dict.items())))
    for i, (c, _) in enumerate(dataset.train_data_local_dict.items()):
        client = FedAvgHam10kClient(str(c), context.model_args, context, dataset.train_data_local_dict[c],
                                    dataset.data_local_train_num_dict[c], dataset.test_data_local_dict[c],
                                    dataset.data_local_test_num_dict[c], context.experiment_logger)
        client.overwrite_model_state(initial_model_state)
        clients.append(client)
        if (i + 1) % 50 == 0:
            logger.debug('... created {}/{}'.format(i + 1, len(dataset.train_data_local_dict.items())))
    return clients
