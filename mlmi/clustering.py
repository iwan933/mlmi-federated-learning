from typing import Dict, List
import random
import logging

import numpy as np
import matplotlib.pyplot as plt
import time

import sklearn
import torch
from torch.utils import data
from torch import Tensor

from mlmi.participant import BaseParticipant

import scipy.cluster.hierarchy as hac


def flatten_model_parameter(state_dict: Dict[str, Tensor], sorted_keys: List[str]) -> Tensor:
    model_state_layer_flatten = torch.cat([torch.flatten(state_dict[k]) for k in sorted_keys if k != 'criterion.weight'])
    return model_state_layer_flatten


def find_nearest(array, id_client, i):
    array = np.asarray(array)
    idx = np.argsort((np.abs(array - array[id_client])))[i]
    return idx


class BaseClusterPartitioner(object):

    def cluster(self, participants: List['BaseParticipant'], server: BaseParticipant) -> Dict[str, List['BaseParticipant']]:
        raise NotImplementedError()


class RandomClusterPartitioner(BaseClusterPartitioner):

    def cluster(self, participants: List['BaseParticipant'], server: BaseParticipant) -> Dict[str, List['BaseParticipant']]:
        num_cluster = 10
        result_dic = {}
        for id in range(1, num_cluster+1):
            result_dic[str(id)] = []
        for participant in participants:
            participant.cluster_id = str(random.randint(1, num_cluster))
            result_dic[participant.cluster_id].append(participant)
        return result_dic


class GradientClusterPartitioner(BaseClusterPartitioner):

    def __init__(self, linkage_mech, criterion, dis_metric, max_value_criterion, plot_dendrogram, reallocate_clients,
                 threshold_min_client_cluster):
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

    def cluster(self, participants: List['BaseParticipant'], server: BaseParticipant) -> Dict[str, List['BaseParticipant']]:
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

    def __init__(self, linkage_mech, criterion, dis_metric, max_value_criterion, plot_dendrogram, reallocate_clients,
                 threshold_min_client_cluster):
        self.linkage_mech = linkage_mech
        self.criterion = criterion
        self.dis_metric = dis_metric
        self.max_value_criterion = max_value_criterion
        self.plot_dendrogram = plot_dendrogram
        self.reallocate_clients = reallocate_clients
        self.threshold_min_client_cluster = threshold_min_client_cluster

    def cluster(self, participants: List['BaseParticipant'], server: BaseParticipant) -> Dict[str, List['BaseParticipant']]:
        logging.info('start clustering...')
        clusters_hac_dic = {}

        # Compute distance matrix of model updates: Using mean of weights from last layer of each participant
        model_states: List[Dict[str, Tensor]] = [p.model.state_dict() for p in participants]
        keys = list(model_states[0].keys())
        # to flatten models without bias use version below
        # keys = list(filter(lambda k: not k.endswith('bias'), model_states[0].keys()))
        model_parameter = np.array([flatten_model_parameter(m, keys).numpy() for m in model_states], dtype=float)

        tic = time.perf_counter()
        cluster_ids = hac.fclusterdata(model_parameter, self.max_value_criterion, self.criterion,
                                       method=self.linkage_mech, metric=self.dis_metric)
        toc = time.perf_counter()
        print(f'Computation time:{toc-tic}')

        num_cluster = max(cluster_ids)

        # Allocate participants to clusters
        i = 0
        for id in range(1, num_cluster + 1):
            clusters_hac_dic[str(id)] = []
        for participant in participants:
            participant.cluster_id = str(cluster_ids[i])
            clusters_hac_dic[participant.cluster_id].append(participant)
            i += 1

        if self.reallocate_clients:
            logging.info('Start reallocating lonely clients')
            logging.info(f'Initially found {num_cluster} clusters')
            server: Dict[str, Tensor] = server.model.state_dict()
            global_parameter = flatten_model_parameter(server, keys).cpu().numpy()
            euclidean_dist = np.array([((model_parameter[participant_id] - global_parameter) ** 2).sum(axis=0)
                                       for participant_id in range(len(participants))])

            lonely_clusters_id = []
            cluster_ids_arr = np.asarray(cluster_ids)
            for cluster_id in range(num_cluster):
                if np.count_nonzero(cluster_ids_arr == cluster_id + 1) <= self.threshold_min_client_cluster:
                    logging.info('cluster {} is under the minimal client threshold'.format(cluster_id + 1))
                    lonely_clusters_id.append(cluster_id + 1)

            empty_cluster_id = []
            nearest_cluster_id = None
            for lonely_cluster_id in lonely_clusters_id:
                i = 1
                if len(clusters_hac_dic[str(lonely_cluster_id)]) > self.threshold_min_client_cluster:
                    pass
                else:
                    # reallocate lonely client to nearest cluster
                    lonely_clients = clusters_hac_dic[str(lonely_cluster_id)]
                    id_clients = np.where(cluster_ids == lonely_cluster_id)[0]
                    for k, id_client in enumerate(id_clients):
                        while nearest_cluster_id in empty_cluster_id or nearest_cluster_id == lonely_cluster_id or i == 1:
                            nearest_client_id = find_nearest(euclidean_dist, id_client, i)
                            nearest_cluster_id = cluster_ids[nearest_client_id]
                            i += 1
                        clusters_hac_dic[str(nearest_cluster_id)].append(lonely_clients[k])
                        cluster_ids[id_client] = nearest_cluster_id
                    clusters_hac_dic[str(lonely_cluster_id)] = {}
                    empty_cluster_id.append(lonely_cluster_id)

            for key in empty_cluster_id:
                del clusters_hac_dic[str(key)]
            num_cluster = num_cluster - len(empty_cluster_id)

        logging.info(f'Final cluster number:{num_cluster}')
        clusters_hac_dic_new = {str(i + 1): val for i, (key, val) in enumerate(clusters_hac_dic.items())}

        logging.info('Used linkage method: ' + str(self.linkage_mech))
        logging.info('Used distance method: ' + str(self.dis_metric))
        logging.info('Used criterion for clustering: ' + str(self.criterion))
        logging.info('Found %i clusters', num_cluster)
        logging.info('Finished clustering')

        return clusters_hac_dic_new


class AlternativePartitioner(BaseClusterPartitioner):

    def __init__(self, linkage_mech, criterion, dis_metric, max_value_criterion, plot_dendrogram, reallocate_clients,
                 threshold_min_client_cluster):

        self.linkage_mech = linkage_mech
        self.criterion = criterion
        self.dis_metric = dis_metric
        self.max_value_criterion = max_value_criterion
        self.plot_dendrogram = plot_dendrogram
        self.reallocate_clients = reallocate_clients
        self.threshold_min_client_cluster = threshold_min_client_cluster

    def cluster(self, participants: List['BaseParticipant'], server) -> Dict[str, List['BaseParticipant']]:
        logging.info('start clustering...')
        clusters_hac_dic = {}
        server: Dict[str, Tensor] = server.model.state_dict()

        model_states: List[Dict[str, Tensor]] = [p.model.state_dict() for p in participants]
        keys = list(model_states[0].keys())
        model_parameter = np.array([flatten_model_parameter(m, keys).numpy() for m in model_states], dtype=float)

        global_parameter = flatten_model_parameter(server, keys).cpu().numpy()
        euclidean_dist = np.array([((model_parameter[participant_id]-global_parameter)**2).sum(axis=0)
                                   for participant_id in range(len(participants))])

        cluster_ids = hac.fclusterdata(np.reshape(euclidean_dist, (len(euclidean_dist), 1)), self.max_value_criterion,
                                       self.criterion, method=self.linkage_mech, metric=self.dis_metric)

        # Allocate participants to clusters
        i = 0
        num_cluster = max(cluster_ids)
        for id in range(1, num_cluster + 1):
            clusters_hac_dic[str(id)] = []
        for participant in participants:
            participant.cluster_id = str(cluster_ids[i])
            clusters_hac_dic[participant.cluster_id].append(participant)
            i += 1

        if self.reallocate_clients:
            logging.info('Start reallocating lonely clients')
            logging.info(f'Initially found {num_cluster} clusters')

            lonely_clusters_id = []
            cluster_ids_arr = np.asarray(cluster_ids)
            for cluster_id in range(num_cluster):
                if np.count_nonzero(cluster_ids_arr == cluster_id+1) <= self.threshold_min_client_cluster:
                    logging.info('cluster {} is under the minimal client threshold'.format(cluster_id + 1))
                    lonely_clusters_id.append(cluster_id+1)

            empty_cluster_id = []
            nearest_cluster_id = None
            for lonely_cluster_id in lonely_clusters_id:
                i = 1
                if len(clusters_hac_dic[str(lonely_cluster_id)]) > self.threshold_min_client_cluster:
                    pass
                else:
                    # reallocate lonely client to nearest cluster
                    lonely_clients = clusters_hac_dic[str(lonely_cluster_id)]
                    id_clients = np.where(cluster_ids == lonely_cluster_id)[0]
                    for k, id_client in enumerate(id_clients):
                        while nearest_cluster_id in empty_cluster_id or nearest_cluster_id == lonely_cluster_id or i == 1:
                            nearest_client_id = find_nearest(euclidean_dist, id_client, i)
                            nearest_cluster_id = cluster_ids[nearest_client_id]
                            i += 1
                        clusters_hac_dic[str(nearest_cluster_id)].append(lonely_clients[k])
                        cluster_ids[id_client] = nearest_cluster_id
                    clusters_hac_dic[str(lonely_cluster_id)] = {}
                    empty_cluster_id.append(lonely_cluster_id)

            for key in empty_cluster_id:
                del clusters_hac_dic[str(key)]
            num_cluster = num_cluster - len(empty_cluster_id)

        logging.info(f'Final cluster number:{num_cluster}')
        clusters_hac_dic_new = {str(i + 1): val for i, (key, val) in enumerate(clusters_hac_dic.items())}

        logging.info('Used linkage method: ' + str(self.linkage_mech))
        logging.info('Used distance method: ' + str(self.dis_metric))
        logging.info('Used criterion for clustering: ' + str(self.criterion))
        logging.info('Found %i clusters', num_cluster)
        logging.info('Finished clustering')
        return clusters_hac_dic_new


class FixedAlternativePartitioner(BaseClusterPartitioner):

    def __init__(self, linkage_mech, criterion, dis_metric, max_value_criterion, plot_dendrogram, reallocate_clients,
                 threshold_min_client_cluster):

        self.linkage_mech = linkage_mech
        self.criterion = criterion
        self.dis_metric = dis_metric
        self.max_value_criterion = max_value_criterion
        self.plot_dendrogram = plot_dendrogram
        self.reallocate_clients = reallocate_clients
        self.threshold_min_client_cluster = threshold_min_client_cluster

    def cluster(self, participants: List['BaseParticipant'], server) -> Dict[str, List['BaseParticipant']]:
        logging.info('start clustering...')
        clusters_hac_dic = {}
        server: Dict[str, Tensor] = server.model.state_dict()

        model_states: List[Dict[str, Tensor]] = [p.model.state_dict() for p in participants]
        keys = list(model_states[0].keys())
        model_parameter = np.array([flatten_model_parameter(m, keys).numpy() for m in model_states], dtype=float)

        global_parameter = flatten_model_parameter(server, keys).cpu().numpy()
        euclidean_dist = np.array([(((model_parameter[participant_id]-global_parameter)**2).sum(axis=0) ** (1/2))
                                   for participant_id in range(len(participants))])

        cluster_ids = hac.fclusterdata(np.reshape(euclidean_dist, (len(euclidean_dist), 1)), self.max_value_criterion,
                                       self.criterion, method=self.linkage_mech, metric=self.dis_metric)

        # Allocate participants to clusters
        i = 0
        num_cluster = max(cluster_ids)
        for id in range(1, num_cluster + 1):
            clusters_hac_dic[str(id)] = []
        for participant in participants:
            participant.cluster_id = str(cluster_ids[i])
            clusters_hac_dic[participant.cluster_id].append(participant)
            i += 1

        if self.reallocate_clients:
            logging.info('Start reallocating lonely clients')
            logging.info(f'Initially found {num_cluster} clusters')

            lonely_clusters_id = []
            cluster_ids_arr = np.asarray(cluster_ids)
            for cluster_id in range(num_cluster):
                if np.count_nonzero(cluster_ids_arr == cluster_id+1) <= self.threshold_min_client_cluster:
                    logging.info('cluster {} is under the minimal client threshold'.format(cluster_id + 1))
                    lonely_clusters_id.append(cluster_id+1)

            empty_cluster_id = []
            nearest_cluster_id = None
            for lonely_cluster_id in lonely_clusters_id:
                i = 1
                if len(clusters_hac_dic[str(lonely_cluster_id)]) > self.threshold_min_client_cluster:
                    pass
                else:
                    # reallocate lonely client to nearest cluster
                    lonely_clients = clusters_hac_dic[str(lonely_cluster_id)]
                    id_clients = np.where(cluster_ids == lonely_cluster_id)[0]
                    for k, id_client in enumerate(id_clients):
                        while nearest_cluster_id in empty_cluster_id or nearest_cluster_id == lonely_cluster_id or i == 1:
                            nearest_client_id = find_nearest(euclidean_dist, id_client, i)
                            nearest_cluster_id = cluster_ids[nearest_client_id]
                            i += 1
                        clusters_hac_dic[str(nearest_cluster_id)].append(lonely_clients[k])
                        cluster_ids[id_client] = nearest_cluster_id
                    clusters_hac_dic[str(lonely_cluster_id)] = {}
                    empty_cluster_id.append(lonely_cluster_id)

            for key in empty_cluster_id:
                del clusters_hac_dic[str(key)]
            num_cluster = num_cluster - len(empty_cluster_id)

        logging.info(f'Final cluster number:{num_cluster}')
        clusters_hac_dic_new = {str(i + 1): val for i, (key, val) in enumerate(clusters_hac_dic.items())}

        logging.info('Used linkage method: ' + str(self.linkage_mech))
        logging.info('Used distance method: ' + str(self.dis_metric))
        logging.info('Used criterion for clustering: ' + str(self.criterion))
        logging.info('Found %i clusters', num_cluster)
        logging.info('Finished clustering')
        return clusters_hac_dic_new


class DatadependentPartitioner(BaseClusterPartitioner):

    def __init__(
            self,
            dataloader: torch.utils.data.DataLoader,
            linkage_mech,
            criterion,
            dis_metric,
            max_value_criterion,
            threshold_min_client_cluster,
            *args,
            **kwargs
    ):
        self.linkage_mech = linkage_mech
        self.criterion = criterion
        self.dis_metric = dis_metric
        self.max_value_criterion = max_value_criterion
        self.dataloader = dataloader

    def predict(self, participant: BaseParticipant):
        predictions = np.array([], dtype=np.float)
        model = participant.model.cpu()
        for x, y in self.dataloader:
            x = x.cpu()
            y = y.cpu()
            logits = model.model(x)
            prob, idx = torch.max(logits, dim=1)
            correct = np.zeros((*idx.shape,))
            correct[idx.numpy() == y.numpy()] = 1
            predictions = np.append(predictions, correct)
            # label and probability version:
            # predictions = np.append(predictions, np.array(list(zip(prob.cpu(),idx.cpu()))))
        return predictions

    def cluster(
            self,
            participants: List['BaseParticipant'],
            server: BaseParticipant
    ) -> Dict[str, List['BaseParticipant']]:
        model_predictions = np.array([self.predict(p) for p in participants])
        cluster_ids = hac.fclusterdata(model_predictions, self.max_value_criterion, self.criterion,
                                       method=self.linkage_mech, metric=self.dis_metric)

        num_cluster = max(cluster_ids)

        # Allocate participants to clusters
        i = 0
        clusters_hac_dic = {}
        for id in range(1, num_cluster + 1):
            clusters_hac_dic[str(id)] = []
        for participant in participants:
            participant.cluster_id = str(cluster_ids[i])
            clusters_hac_dic[participant.cluster_id].append(participant)
            i += 1
        return clusters_hac_dic
