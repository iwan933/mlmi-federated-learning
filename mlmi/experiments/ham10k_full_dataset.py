import pytorch_lightning as pl
import torch

from sacred import Experiment

from mlmi.datasets.ham10k import load_ham10k_few_big_many_small_federated2fulldataset
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
    lr = 0.001
    batch_size = 16
    epochs = 64  # 210 for 2400, 840 for 10000


@ex.automain
def run_full_dataset(seed, lr, batch_size, epochs):
    fix_random_seeds(seed)

    train_dataloader, test_dataloader = load_ham10k_few_big_many_small_federated2fulldataset()
    optimizer_args = OptimizerArgs(
        optimizer_class=torch.optim.SGD,
        lr=lr
    )
    model = MobileNetV2Lightning(num_classes=7, participant_name='full', optimizer_args=optimizer_args, pretrain=False)
    logger = create_tensorboard_logger('ham10kmobilenetv2')
    test_each_x_epochs = 10

    for i in range(0, epochs):
        trainer = pl.Trainer(logger=False, checkpoint_callback=False, gpus=1, min_steps=1, max_steps=1,
                             progress_bar_refresh_rate=0)
        log.info(f'starting epoch {i}')
        trainer.fit(model, train_dataloader)
        save_full_state(model.state_dict(), i + 1, lr, batch_size)
        GlobalConfusionMatrix().enable_logging()
        result = trainer.test(model, test_dataloader)[0]
        GlobalConfusionMatrix().disable_logging()
        loss = result.get('test/loss/full')
        acc = result.get('test/acc/full')
        balanced_acc = result.get('test/balanced_acc/full')

        log_loss_and_acc('global-train-test', torch.FloatTensor([loss]), torch.FloatTensor([acc]), logger, i + 1)
        log_loss_and_acc('global-balanced-train-test', torch.FloatTensor([loss]), torch.FloatTensor([balanced_acc]),
                         logger, i + 1)

        try:
            global_confusion_matrix = GlobalConfusionMatrix()
            if global_confusion_matrix.has_data:
                matrix = global_confusion_matrix.compute()
                image = generate_confusion_matrix_heatmap(matrix)
                logger.experiment.add_image('test/confusionmatrix', image.numpy(), i + 1)
        except Exception as e:
            print('failed to log confusion matrix (global)', e)
    GlobalConfusionMatrix().enable_logging()
    trainer = pl.Trainer(logger=False, checkpoint_callback=False, gpus=1, min_epochs=1, max_epochs=1)
    result = trainer.test(model, test_dataloader)[0]
    GlobalConfusionMatrix().disable_logging()
    loss = result.get('test/loss/full')
    acc = result.get('test/acc/full')
    balanced_acc = result.get('test/balanced_acc/full')
    log_loss_and_acc('global-train-test', loss, acc, logger, epochs)
    log_loss_and_acc('global-balanced-train-test', loss, balanced_acc, logger, epochs)
    try:
        global_confusion_matrix = GlobalConfusionMatrix()
        if global_confusion_matrix.has_data:
            matrix = global_confusion_matrix.compute()
            image = generate_confusion_matrix_heatmap(matrix)
            logger.experiment.add_image('test/confusionmatrix', image.numpy(), epochs)
    except Exception as e:
        print('failed to log confusion matrix (global)', e)
