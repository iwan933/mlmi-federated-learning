import sys

from mlmi.models.ham10k import GlobalConfusionMatrix, GlobalTestTestConfusionMatrix, GlobalTrainTestConfusionMatrix

sys.path.append('C:/Users/Richard/Desktop/Informatik/Semester_5/MLMI/git/mlmi-federated-learning')

from mlmi.structs import FederatedDatasetData, ModelArgs

import random
from mlmi.log import getLogger
from mlmi.reptile.model import ReptileClient, ReptileServer
from mlmi.reptile.util import reptile_train_step
from mlmi.reptile.structs import ReptileExperimentContext
from mlmi.settings import REPO_ROOT
from mlmi.utils import evaluate_local_models
from mlmi.fedavg.data import swap_labels


logger = getLogger(__name__)


def cyclerange(start, interval, total_len):
    assert start < total_len, "Error: start must be < total_len"
    if interval >= total_len:
        return list(range(start, total_len)) + list(range(0, start))
    if start + interval > total_len:
        return list(range(start, total_len)) + list(range(0, (start + interval) % total_len))
    return list(range(start, start + interval))

def initialize_clients(dataset: FederatedDatasetData,
                       model_args: ModelArgs,
                       context,
                       experiment_logger):
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
            lightning_logger=experiment_logger
        )
        clients.append(client)
    return clients


def run_reptile(context: ReptileExperimentContext,
                dataset_train: FederatedDatasetData,
                dataset_test: FederatedDatasetData,
                initial_model_state,
                after_round_evaluation):
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
        experiment_logger=context.experiment_logger
    )
    test_clients = None
    if dataset_test is not None:
        test_clients = initialize_clients(
            dataset=dataset_test,
            model_args=context.inner_model_args,
            context=context.name,
            experiment_logger=context.experiment_logger
        )

    # Set up server
    server = ReptileServer(
        participant_name='initial_server',
        model_args=context.meta_model_args,
        context=context.name,
        initial_model_state=initial_model_state
    )

    # Perform training
    for i in range(context.num_meta_steps):
        if context.meta_batch_size == -1:
            meta_batch = train_clients
        else:
            meta_batch = [
                train_clients[k] for k in cyclerange(
                    start=i*context.meta_batch_size % len(train_clients),
                    interval=context.meta_batch_size,
                    total_len=len(train_clients)
                )
            ]
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
            # Pick train / test clients at random and test on them
            losses, accs, balanced_accs = [], [], []
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
                    result = evaluate_local_models(participants=clients)
                    losses.append(result.get('test/loss'))
                    accs.append(result.get('test/acc'))
                    balanced_accs.append(result.get('test/balanced_acc'))
                else:
                    losses.append(None)
                    accs.append(None)
                    balanced_accs.append(None)

            # Log
            if after_round_evaluation is not None:
                for c in after_round_evaluation:
                    c('', losses[0], accs[0], balanced_accs[0], losses[1], accs[1], balanced_accs[1], i)

        logger.info('finished training round')

    if context.do_final_evaluation:
        # Final evaluation on subsample of train / test clients
        losses, accs, balanced_accs = [], [], []
        is_train_client_set = True
        for client_set in [train_clients, test_clients]:
            if client_set:
                if context.num_eval_clients_final == -1:
                    eval_clients = client_set
                else:
                    eval_clients = RANDOM.sample(
                        client_set, context.num_eval_clients_final
                    )
                reptile_train_step(
                    aggregator=server,
                    participants=eval_clients,
                    inner_training_args=context.get_inner_training_args(eval=True),
                    evaluation_mode=True
                )
                if is_train_client_set:
                    GlobalTrainTestConfusionMatrix().enable_logging()
                else:
                    GlobalTestTestConfusionMatrix().enable_logging()
                result = evaluate_local_models(participants=eval_clients)
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
                c('final_', losses[0], accs[0], balanced_accs[0], losses[1], accs[1], balanced_accs[1], 0)
