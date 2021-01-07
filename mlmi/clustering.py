from typing import Dict, List, TypeVar
import random

from mlmi.participant import BaseParticipant
from mlmi.struct import ClusterArgs

import scipy.cluster.hierarchy as hac
from torch import mean
import numpy as np
import matplotlib.pyplot as plt


T = TypeVar('T', bound=BaseParticipant)


class BaseClusterPartitioner(object):

    def cluster(self, participants: List[T]) -> Dict[str, List[T]]:
        raise NotImplementedError()


class RandomClusterPartitioner(BaseClusterPartitioner):

    def cluster(self, participants: List[T]) -> Dict[str, List[T]]:
        num_cluster = 10
        result_dic = {}
        for id in range(1, num_cluster+1):
            result_dic[str(id)] = []
        for participant in participants:
            participant.cluster_id = str(random.randint(1, num_cluster))
            result_dic[participant.cluster_id].append(participant)
        return result_dic


class GradientClusterPartitioner(BaseClusterPartitioner):

    def __init__(self, linkage_mech, criterion, dis_metric, max_value_criterion):
        self.linkage_mech = linkage_mech
        self.criterion = criterion
        self.dis_metric = dis_metric
        self.max_value_criterion = max_value_criterion

    def cluster(self, participants: List[BaseParticipant]) -> Dict[str, List[BaseParticipant]]:
        clusters_hac_dic = {}

        # Compute distance matrix of model updates: Using mean of weights from last layer of each participant
        model_updates = np.array([])
        for participant in participants:
            weights_last_layer_key = list(participant.model.state_dict().keys())[-2]
            weights_last_layer = participant.model.state_dict()[weights_last_layer_key]
            model_updates = np.append(model_updates, mean(weights_last_layer).numpy())
        model_updates = np.reshape(model_updates, (len(model_updates), 1))
        distance_matrix = hac.linkage(model_updates, method=self.linkage_mech, metric=self.dis_metric, optimal_ordering=False)

        # Alternative
        #cluster_ids_alt = hac.fclusterdata(model_updates, 4, criterion="maxclust", metric="euclidean", method="single")

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

