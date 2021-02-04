from typing import Callable, Dict, List, Optional

from torch import Tensor

from mlmi.fedavg.util import evaluate_cluster_models, load_fedavg_hierarchical_cluster_configuration
from mlmi.log import getLogger
from mlmi.participant import BaseAggregatorParticipant, BaseParticipant, BaseTrainingParticipant
from mlmi.structs import ClusterArgs, FederatedDatasetData
from mlmi.utils import evaluate_global_model

logger = getLogger(__name__)


def run_fedavg_hierarchical(
        server: 'BaseAggregatorParticipant',
        clients: List['BaseTrainingParticipant'],
        cluster_args: 'ClusterArgs',
        initial_train_fn: Callable[[List['BaseParticipant']], List['BaseParticipant']],
        federated_round_fn: Callable[['BaseAggregatorParticipant', List['BaseTrainingParticipant']], any],
        create_aggregator_fn: Callable[[str], 'BaseAggregatorParticipant'],
        after_post_clustering_evaluation: Optional[List[Callable[[Tensor, Tensor, int], None]]] = None,
        after_clustering_round_evaluation: Optional[List[Callable[[str, Tensor, Tensor, int], None]]] = None,
        after_federated_round_evaluation: Optional[List[Callable[[Tensor, Tensor, int], None]]] = None,
        after_clustering: Optional[List[Callable[[Dict[str, List['BaseTrainingParticipant']]], None]]] = None
):
    num_rounds_init = cluster_args.num_rounds_init
    num_rounds_cluster = cluster_args.num_rounds_cluster
    logger.debug('starting local training before clustering.')
    trained_participants = initial_train_fn(clients)
    if len(trained_participants) != len(clients):
        raise ValueError('not all clients successfully participated in the clustering round')

    # Clustering of participants by model updates
    partitioner = cluster_args()
    cluster_clients_dic = partitioner.cluster(clients, server)
    _cluster_clients_dic = dict()
    for cluster_id, participants in cluster_clients_dic.items():
        _cluster_clients_dic[cluster_id] = [c._name for c in participants]

    # Initialize cluster models
    cluster_server_dic = {}
    for cluster_id, participants in cluster_clients_dic.items():
        cluster_server = create_aggregator_fn('cluster_server' + cluster_id)
        cluster_server.aggregate(participants)
        cluster_server_dic[cluster_id] = cluster_server

    if after_clustering is not None:
        for c in after_clustering:
            c(cluster_clients_dic)

    eval_result = evaluate_global_model(global_model_participant=server, participants=clients)
    loss, acc = eval_result.get('test/loss'), eval_result.get('test/acc')
    if after_post_clustering_evaluation is not None:
        for c in after_post_clustering_evaluation:
            c(loss, acc, num_rounds_init)
    global_losses, global_acc = evaluate_cluster_models(cluster_server_dic, cluster_clients_dic)
    if after_post_clustering_evaluation is not None:
        for c in after_post_clustering_evaluation:
            c(global_losses, global_acc, num_rounds_init + 1)

    # Train in clusters
    for i in range(num_rounds_cluster):
        for cluster_id in cluster_clients_dic.keys():
            # train
            cluster_server = cluster_server_dic[cluster_id]
            cluster_clients = cluster_clients_dic[cluster_id]
            logger.info(f'starting training cluster {cluster_id} in round {i + 1}')
            # execute federated learning round
            federated_round_fn(cluster_server, cluster_clients)
            # test
            result = evaluate_global_model(global_model_participant=cluster_server, participants=cluster_clients)
            loss, acc = result.get('test/loss'), result.get('test/acc')
            # log after cluster round
            if after_clustering_round_evaluation is not None:
                for c in after_clustering_round_evaluation:
                    c(f'cluster{cluster_id}', loss, acc, num_rounds_init + i)
            logger.info(f'finished training cluster {cluster_id}')
        logger.info('testing clustering round results')
        global_losses, global_acc = evaluate_cluster_models(cluster_server_dic, cluster_clients_dic)
        # log after full round over all clusters
        if after_federated_round_evaluation is not None:
            for c in after_federated_round_evaluation:
                c(global_losses, global_acc, num_rounds_init + i)
