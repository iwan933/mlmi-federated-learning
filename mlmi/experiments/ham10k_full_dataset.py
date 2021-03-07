import pytorch_lightning as pl
import torch

from sacred import Experiment

from mlmi.datasets.ham10k import load_ham10k_few_big_many_small_federated2fulldataset
from mlmi.models.ham10k import GlobalConfusionMatrix, MobileNetV2Lightning
from mlmi.plot import generate_confusion_matrix_heatmap
from mlmi.structs import OptimizerArgs
from mlmi.utils import create_tensorboard_logger

ex = Experiment('Ham10k full dataset')


@ex.config
def DefaultConfig():
    lr = 0.001
    batch_size = 8
    epochs = 150


@ex.automain
def run_full_dataset(lr, batch_size, epochs):
    train_dataloader, test_dataloader = load_ham10k_few_big_many_small_federated2fulldataset()
    optimizer_args = OptimizerArgs(
        optimizer_class=torch.optim.SGD,
        lr=lr
    )
    model = MobileNetV2Lightning(num_classes=7, participant_name='full', optimizer_args=optimizer_args)
    logger = create_tensorboard_logger('ham10kmobilenetv2')
    test_each_x_epochs = 10

    for i in range(epochs):
        trainer = pl.Trainer(gpus=1, min_epochs=1, max_epochs=1)
        trainer.fit(model, train_dataloader)
        if i % test_each_x_epochs == 0:
            GlobalConfusionMatrix().enable_logging()
            result = trainer.test(model, test_dataloader)[0]
            GlobalConfusionMatrix().disable_logging()
            loss = result.get('test/loss/full')
            acc = result.get('test/acc/full')
            balanced_acc = result.get('test/balanced_acc/full')
            logger.experiment.add_scalar('test/loss', loss, i)
            logger.experiment.add_scalar('test/acc', acc, i)
            logger.experiment.add_scalar('test/balance-acc', balanced_acc, i)
            try:
                global_confusion_matrix = GlobalConfusionMatrix()
                if global_confusion_matrix.has_data:
                    matrix = global_confusion_matrix.compute()
                    image = generate_confusion_matrix_heatmap(matrix)
                    logger.experiment.add_image('test/confusionmatrix', image.numpy(), i)
            except Exception as e:
                print('failed to log confusion matrix (global)', e)
    GlobalConfusionMatrix().enable_logging()
    trainer = pl.Trainer(gpus=1, min_epochs=1, max_epochs=1)
    result = trainer.test(model, test_dataloader)[0]
    GlobalConfusionMatrix().disable_logging()
    loss = result.get('test/loss/full')
    acc = result.get('test/acc/full')
    balanced_acc = result.get('test/balanced_acc/full')
    logger.experiment.add_scalar('test/loss', loss, i)
    logger.experiment.add_scalar('test/acc', acc, i)
    logger.experiment.add_scalar('test/balance-acc', balanced_acc, i)
    try:
        global_confusion_matrix = GlobalConfusionMatrix()
        if global_confusion_matrix.has_data:
            matrix = global_confusion_matrix.compute()
            image = generate_confusion_matrix_heatmap(matrix)
            logger.experiment.add_image('test/confusionmatrix', image.numpy(), i)
    except Exception as e:
        print('failed to log confusion matrix (global)', e)
