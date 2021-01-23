from typing import Dict, List
import random
import logging

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import Tensor

from mlmi.participant import BaseParticipant

import scipy.cluster.hierarchy as hac


def flatten_model_parameter(state_dict: Dict[str, Tensor], sorted_keys: List[str]) -> Tensor:
    model_state_layer_flatten = torch.cat([torch.flatten(state_dict[k]) for k in sorted_keys])
    return model_state_layer_flatten


class BaseClusterPartitioner(object):

    def cluster(self, participants: List['BaseParticipant']) -> Dict[str, List['BaseParticipant']]:
        raise NotImplementedError()


class RandomClusterPartitioner(BaseClusterPartitioner):

    def cluster(self, participants: List['BaseParticipant']) -> Dict[str, List['BaseParticipant']]:
        num_cluster = 10
        result_dic = {}
        for id in range(1, num_cluster+1):
            result_dic[str(id)] = []
        for participant in participants:
            participant.cluster_id = str(random.randint(1, num_cluster))
            result_dic[participant.cluster_id].append(participant)
        return result_dic


class GradientClusterPartitioner(BaseClusterPartitioner):

    def __init__(self, linkage_mech, criterion, dis_metric, max_value_criterion, plot_dendrogram):
        self.linkage_mech = linkage_mech
        self.criterion = criterion
        self.dis_metric = dis_metric
        self.max_value_criterion = max_value_criterion
        self.plot_dendrogram = plot_dendrogram

    @staticmethod
    def model_weights_mean(participant: 'BaseParticipant'):
        key_layers_participant = list(participant.model.state_dict().keys())
        num_layers = int(len(participant.model.state_dict().keys()) / 2)
        mean_weights_participant = 0
        for layer in range(num_layers):
            layer_dim = participant.model.state_dict()[key_layers_participant[layer*2]].squeeze().dim()
            weights_layer = participant.model.state_dict()[key_layers_participant[layer * 2]].squeeze()
            mean_weights_layer = weights_layer.mean(tuple(range(layer_dim)))
            mean_weights_layer = float(mean_weights_layer)
            mean_weights_participant = mean_weights_participant + mean_weights_layer
        return mean_weights_participant

    @staticmethod
    def model_weights_sum(participant: 'BaseParticipant'):
        sum_weights_participant = 0
        key_layers_participant = list(participant.model.state_dict().keys())
        num_layers = int(len(participant.model.state_dict().keys()) / 2)
        for layer in range(num_layers):
            weights_layer = participant.model.state_dict()[key_layers_participant[layer * 2]].squeeze()
            sum_weights_participant += float(weights_layer.sum())
        return sum_weights_participant

    def cluster(self, participants: List['BaseParticipant']) -> Dict[str, List['BaseParticipant']]:
        logging.info('Start clustering')
        clusters_hac_dic = {}

        # Compute distance matrix of model updates: Using mean of weights from last layer of each participant
        model_updates = np.array([])
        for participant in participants:
            weights_participant = self.model_weights_sum(participant)
            model_updates = np.append(model_updates, weights_participant)
        model_updates = np.reshape(model_updates, (len(model_updates), 1))
        distance_matrix = hac.linkage(model_updates, method=self.linkage_mech, metric=self.dis_metric, optimal_ordering=False)

        # Compute clusters based on distance matrix
        cluster_ids = hac.fcluster(distance_matrix, self.max_value_criterion, self.criterion)
        num_cluster = max(cluster_ids)

        # Allocate participants to clusters
        i = 0
        for id in range(1, num_cluster + 1):
            clusters_hac_dic[str(id)] = []
        for participant in participants:
            participant.cluster_id = str(cluster_ids[i])
            clusters_hac_dic[participant.cluster_id].append(participant)
            i += 1

        for cluster_id in range(num_cluster):
            logging.info(f'cluster {cluster_id} has {np.count_nonzero(cluster_ids == cluster_id)} clients')
            if np.count_nonzero(cluster_ids == cluster_id) == 1:
                logging.info('cluster {} has only one client!'.format(cluster_id))

        logging.info('Used linkage method: ' + str(self.linkage_mech))
        logging.info('Used distance method: ' + str(self.dis_metric))
        logging.info('Used criterion for clustering: ' + str(self.criterion))
        logging.info('Found %i clusters', num_cluster)
        logging.info('Finished clustering')

        if self.plot_dendrogram:
            # Plotting dendrogram for client clusters
            hac.dendrogram(distance_matrix, leaf_rotation=45., leaf_font_size=12, show_contracted=True)
            plt.title("Dendrogram: Client clusters")
            plt.ylabel("Distance")
            plt.show()

        return clusters_hac_dic


class ModelFlattenWeightsPartitioner(BaseClusterPartitioner):

    def __init__(self, linkage_mech, criterion, dis_metric, max_value_criterion, plot_dendrogram):
        self.linkage_mech = linkage_mech
        self.criterion = criterion
        self.dis_metric = dis_metric
        self.max_value_criterion = max_value_criterion
        self.plot_dendrogram = plot_dendrogram

    def cluster(self, participants: List['BaseParticipant']) -> Dict[str, List['BaseParticipant']]:
        logging.info('start clustering...')
        clusters_hac_dic = {}

        model_states: List[Dict[str, Tensor]] = [p.model.state_dict() for p in participants]
        keys = list(model_states[0].keys())
        # to flatten models without bias use version below
        # keys = list(filter(lambda k: not k.endswith('bias'), model_states[0].keys()))
        model_parameter = np.array([flatten_model_parameter(m, keys).numpy() for m in model_states], dtype=float)
        cluster_ids = hac.fclusterdata(model_parameter, self.max_value_criterion, self.criterion,
                                       method=self.linkage_mech, metric=self.dis_metric)
        num_cluster = max(cluster_ids)

        # Allocate participants to clusters
        i = 0
        for id in range(1, num_cluster + 1):
            clusters_hac_dic[str(id)] = []
        for participant in participants:
            participant.cluster_id = str(cluster_ids[i])
            clusters_hac_dic[participant.cluster_id].append(participant)
            i += 1

        for cluster_id in range(num_cluster):
            logging.info(f'cluster {cluster_id} has {np.count_nonzero(cluster_ids == cluster_id)} clients')
            if np.count_nonzero(cluster_ids == cluster_id) == 1:
                logging.info('cluster {} has only one client!'.format(cluster_id))

        logging.info('Used linkage method: ' + str(self.linkage_mech))
        logging.info('Used distance method: ' + str(self.dis_metric))
        logging.info('Used criterion for clustering: ' + str(self.criterion))
        logging.info('Found %i clusters', num_cluster)
        logging.info('Finished clustering')

        return clusters_hac_dic
