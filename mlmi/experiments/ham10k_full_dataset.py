import pytorch_lightning as pl
import torch

from sacred import Experiment

from mlmi.datasets.ham10k import load_ham10k_few_big_many_small_federated2fulldataset, \
    load_ham10k_partition_by_two_labels_federated2fulldataset
from mlmi.experiments.log import log_loss_and_acc
from mlmi.log import getLogger
from mlmi.models.ham10k import GlobalConfusionMatrix, MobileNetV2Lightning
from mlmi.plot import generate_confusion_matrix_heatmap
from mlmi.settings import REPO_ROOT
from mlmi.structs import OptimizerArgs
from mlmi.utils import create_tensorboard_logger, fix_random_seeds

ex = Experiment('Ham10k full dataset')
log = getLogger(__name__)


def save_full_state(model_state, epoch, lr, batch_size):
    path = REPO_ROOT / 'run' / 'states' / 'ham10k_fulldataset' / f'mobilenetv2_lr{lr}bs{batch_size}{epoch}.mdl'
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_state, path)


@ex.config
def DefaultConfig():
    seed = 4444
    lr = [0.001, 0.0007, 0.0004]
    batch_size = 16
    epochs = 40  # 210 for 2400, 840 for 10000


@ex.automain
def run_full_dataset(seed, lr, batch_size, epochs):
    fix_random_seeds(seed)

    train_dataloader, validation_dataloader, test_dataloader = load_ham10k_partition_by_two_labels_federated2fulldataset(
        batch_size=batch_size
    )
    for _lr in lr:
        optimizer_args = OptimizerArgs(
            optimizer_class=torch.optim.SGD,
            lr=_lr
        )
        model = MobileNetV2Lightning(num_classes=7, participant_name='full', optimizer_args=optimizer_args,
                                     pretrain=True)
        logger = create_tensorboard_logger('ham10kmobilenetv2', f'lr{_lr}')
        trainer = pl.Trainer(logger=logger, checkpoint_callback=False, gpus=1, min_epochs=epochs,
                             max_epochs=epochs, progress_bar_refresh_rate=0)
        trainer.fit(model, train_dataloader, validation_dataloader)
        GlobalConfusionMatrix().enable_logging()
        trainer = pl.Trainer(logger=False, checkpoint_callback=False, gpus=1, min_epochs=1, max_epochs=1)
        result = trainer.test(model, test_dataloader)[0]
        GlobalConfusionMatrix().disable_logging()
        loss = result.get('test/loss/full')
        acc = result.get('test/acc/full')
        balanced_acc = result.get('test/balanced_acc/full')
        log_loss_and_acc('global-train-test', torch.FloatTensor([loss]), torch.FloatTensor([acc]), logger, epochs)
        log_loss_and_acc('global-balanced-train-test', torch.FloatTensor([loss]), torch.FloatTensor([balanced_acc]),
                         logger, epochs)
        try:
            global_confusion_matrix = GlobalConfusionMatrix()
            if global_confusion_matrix.has_data:
                matrix = global_confusion_matrix.compute()
                image = generate_confusion_matrix_heatmap(matrix)
                logger.experiment.add_image('test/confusionmatrix', image.numpy(), epochs)
        except Exception as e:
            print('failed to log confusion matrix (global)', e)
