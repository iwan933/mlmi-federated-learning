from typing import Dict, List
import random

from mlmi.participant import BaseParticipant
from mlmi.struct import ClusterArgs

import scipy.cluster.hierarchy as hac
import torch
from torch import mean
import numpy as np
import matplotlib.pyplot as plt


class BaseClusterPartitioner(object):

    def cluster(self, participants: List[BaseParticipant]) -> Dict[str, List[BaseParticipant]]:
        raise NotImplementedError()


class RandomClusterPartitioner(BaseClusterPartitioner):

    def cluster(self, participants: List[BaseParticipant]) -> Dict[str, List[BaseParticipant]]:
        num_cluster = 10
        result_dic = {}
        for id in range(1, num_cluster+1):
            result_dic[str(id)] = []
        for participant in participants:
            participant.cluster_id = str(random.randint(1, num_cluster))
            result_dic[participant.cluster_id].append(participant)
        return result_dic


class GradientClusterPartitioner(BaseClusterPartitioner):

    def __init__(self, cluster_args: ClusterArgs):
        assert cluster_args is not None, 'Cluster args are required to perform clustering'

        self.cluster_args = cluster_args
        self.linkage_mech = cluster_args.linkage_mech
        self.criterion = cluster_args.criterion
        self.dis_metric = cluster_args.dis_metric
        self.max_value_criterion = cluster_args.max_value_criterion


    def model_weigths(self, participant: BaseParticipant):
        key_layers_participant = list(participant.get_model().state_dict().keys())
        num_layers = int(len(participant.get_model().state_dict().keys()) / 2)
        accumulated_weights_participant = 0
        for layer in range(num_layers):
            layer_dim = participant.get_model().state_dict()[key_layers_participant[layer*2]].squeeze().dim()
            weights_layer = participant.get_model().state_dict()[key_layers_participant[layer * 2]].squeeze()
            mean_weights_layer = weights_layer.mean(tuple(range(layer_dim)))
            mean_weights_layer = float(mean_weights_layer)
            accumulated_weights_participant = accumulated_weights_participant + mean_weights_layer

        return accumulated_weights_participant

    def cluster(self, participants: List[BaseParticipant]) -> Dict[str, List[BaseParticipant]]:
        clusters_hac_dic = {}

        # Compute distance matrix of model updates: Using mean of weights from last layer of each participant
        model_updates = np.array([])
        for participant in participants:
            accumulated_weights_participant = self.model_weigths(participant)
            model_updates = np.append(model_updates, accumulated_weights_participant)

            """"
            weights_last_layer_key = list(participant.get_model().state_dict().keys())[-2]
            weights_last_layer = participant.get_model().state_dict()[weights_last_layer_key]
            model_updates = np.append(model_updates, mean(weights_last_layer).numpy())
            """
        model_updates = np.reshape(model_updates, (len(model_updates), 1))
        distance_matrix = hac.linkage(model_updates, method=self.linkage_mech, metric=self.dis_metric, optimal_ordering=False)

        # Alternative
        # cluster_ids_alt = hac.fclusterdata(model_updates, 4, criterion="maxclust", metric="euclidean", method="single")

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

        # Plotting dendrogram for client clusters
        hac.dendrogram(distance_matrix, leaf_rotation=45., leaf_font_size=12, show_contracted=True)
        plt.title("Dendrogram: Client clusters")
        plt.ylabel("Distance")
        plt.show()

        return clusters_hac_dic
