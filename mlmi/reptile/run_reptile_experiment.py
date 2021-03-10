import random
from typing import Dict

import torch
from torch import Tensor

from mlmi.log import getLogger
from mlmi.reptile.model import ReptileClient, ReptileServer
from mlmi.reptile.util import reptile_train_step
from mlmi.reptile.structs import ReptileExperimentContext
from mlmi.settings import REPO_ROOT
from mlmi.utils import evaluate_local_models
from mlmi.fedavg.data import swap_labels
from mlmi.models.ham10k import GlobalConfusionMatrix, GlobalTestTestConfusionMatrix, GlobalTrainTestConfusionMatrix
from mlmi.structs import FederatedDatasetData, ModelArgs


logger = getLogger(__name__)


def cyclerange(start, interval, total_len):
    assert start < total_len, "Error: start must be < total_len"
    if interval >= total_len:
        return list(range(start, total_len)) + list(range(0, start))
    if start + interval > total_len:
        return list(range(start, total_len)) + list(range(0, (start + interval) % total_len))
    return list(range(start, start + interval))


def save_reptile_state(reptile_context: 'ReptileExperimentContext', meta_step: int, model_state: Dict[str, Tensor]):
    path = REPO_ROOT / 'run' / 'states' / 'reptile_federated' / f'{reptile_context}r{meta_step}.mdl'
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_state, path)


def initialize_clients(dataset: FederatedDatasetData,
                       model_args: ModelArgs,
                       context,
                       experiment_logger,
                       do_balancing):
    clients = []
    for c in dataset.train_data_local_dict.keys():
        client = ReptileClient(
            client_id=str(c),
            model_args=model_args,
            context=context,
            train_dataloader=dataset.train_data_local_dict[c],
            num_train_samples=dataset.data_local_train_num_dict[c],
            test_dataloader=dataset.test_data_local_dict[c],
            num_test_samples=dataset.data_local_test_num_dict[c],
            lightning_logger=experiment_logger,
            do_balancing=do_balancing
        )
        clients.append(client)
    return clients


def run_reptile(context: ReptileExperimentContext,
                dataset_train: FederatedDatasetData,
                dataset_test: FederatedDatasetData,
                initial_model_state,
                after_round_evaluation,
                start_round
):
    def _evaluate(num_clients: int,
                  tag: str,
                  global_step: int):
        """Evaluate on train and test clients"""

        losses, accs, balanced_accs = [], [], []
        is_train_client_set = True
        for client_set in [train_clients, test_clients]:
            if client_set:
                if context.num_eval_clients_training == -1:
                    clients = client_set
                else:
                    clients = RANDOM.sample(
                        client_set, context.num_eval_clients_training
                    )
                reptile_train_step(
                    aggregator=server,
                    participants=clients,
                    inner_training_args=context.get_inner_training_args(eval=True),
                    evaluation_mode=True
                )
                if is_train_client_set:
                    GlobalTrainTestConfusionMatrix().enable_logging()
                else:
                    GlobalTestTestConfusionMatrix().enable_logging()
                result = evaluate_local_models(participants=clients)
                if is_train_client_set:
                    GlobalTrainTestConfusionMatrix().disable_logging()
                else:
                    GlobalTestTestConfusionMatrix().disable_logging()
                losses.append(result.get('test/loss'))
                accs.append(result.get('test/acc'))
                balanced_accs.append(result.get('test/balanced_acc'))
            else:
                losses.append(None)
                accs.append(None)
                balanced_accs.append(None)
            is_train_client_set = False

        # Log
        if after_round_evaluation is not None:
            for c in after_round_evaluation:
                c(tag, losses[0], accs[0], balanced_accs[0], losses[1], accs[1], balanced_accs[1], global_step)

    RANDOM = random.Random(context.seed)

    # Randomly swap labels
    if context.swap_labels:
        dataset_train = swap_labels(
            fed_dataset=dataset_train,
            max_classes_per_client=62,
            random_seed=context.seed
        )
        if dataset_test is not None:
            dataset_test = swap_labels(
                fed_dataset=dataset_test,
                max_classes_per_client=62,
                random_seed=context.seed
            )

    # Set up clients
    train_clients = initialize_clients(
        dataset=dataset_train,
        model_args=context.inner_model_args,
        context=context.name,
        experiment_logger=context.experiment_logger,
        do_balancing=context.do_balancing
    )
    test_clients = None
    if dataset_test is not None:
        test_clients = initialize_clients(
            dataset=dataset_test,
            model_args=context.inner_model_args,
            context=context.name,
            experiment_logger=context.experiment_logger,
            do_balancing=context.do_balancing
        )

    # Set up server
    server = ReptileServer(
        participant_name='initial_server',
        model_args=context.meta_model_args,
        context=context,
        initial_model_state=initial_model_state
    )

    # Perform training
    for i in range(start_round, context.num_meta_steps):
        if context.meta_batch_size == -1:
            meta_batch = train_clients
        else:
            meta_batch = RANDOM.sample(train_clients, context.meta_batch_size)
        # Meta training step
        reptile_train_step(
            aggregator=server,
            participants=meta_batch,
            inner_training_args=context.get_inner_training_args(),
            meta_training_args=context.get_meta_training_args(
                frac_done=i / context.num_meta_steps
            )
        )

        # Evaluation on train and test clients
        if i % context.eval_interval == 0:
            # Save server model state
            save_reptile_state(context, i, server.model.state_dict())
            # Pick train / test clients at random and test on them
            _evaluate(
                num_clients=context.num_eval_clients_training,
                tag='',
                global_step=i
            )
        logger.info('finished training round')

    if context.do_final_evaluation:
        # Final evaluation on subsample of train / test clients
        _evaluate(
            num_clients=context.num_eval_clients_final,
            tag='final_',
            global_step=0
        )
