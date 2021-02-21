import numpy as np
from typing import Dict, Optional

from torch import Tensor

from mlmi.log import getLogger
from mlmi.participant import BaseTrainingParticipant


logger = getLogger(__name__)


def estimate_weights(labels):
    label, counts = np.unique(labels, return_counts=True)
    return 1 / counts


class FedAvgHam10kClient(BaseTrainingParticipant):

    def get_model_kwargs(self) -> Optional[Dict]:
        labels = np.array([])
        for _, y in self.train_data_loader:
            labels = np.append(labels, y)
        for _, y in self.test_data_loader:
            labels = np.append(labels, y)
        return {
            'weights': estimate_weights(labels)
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
