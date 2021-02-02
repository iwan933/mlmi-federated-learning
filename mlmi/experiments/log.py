import torch
from pytorch_lightning.loggers import LightningLoggerBase


def log_loss_and_acc(model_name: str, loss: torch.Tensor, acc: torch.Tensor, experiment_logger: LightningLoggerBase,
                     global_step: int):
    """
    Logs the loss and accuracy in an histogram as well as scalar
    :param model_name: name for logging
    :param loss: loss tensor
    :param acc: acc tensor
    :param experiment_logger: lightning logger
    :param global_step: global step
    :return:
    """
    experiment_logger.experiment.add_histogram(f'{model_name}/acc/test', acc, global_step=global_step)
    experiment_logger.experiment.add_scalar(f'{model_name}/acc/test/mean', torch.mean(acc),
                                            global_step=global_step)
    if loss.dim() == 0:
        loss = torch.tensor([loss])
    for x in loss:
        if torch.isnan(x) or torch.isinf(x):
            return
    experiment_logger.experiment.add_histogram(f'{model_name}/loss/test/', loss, global_step=global_step)
    experiment_logger.experiment.add_scalar(f'{model_name}/loss/test/mean', torch.mean(loss),
                                            global_step=global_step)


def log_goal_test_acc(model_name: str, acc: torch.Tensor,
                      experiment_logger: LightningLoggerBase, global_step: int):
    if acc.dim() == 0:
        acc = torch.tensor([acc])
    over80 = acc[acc >= 0.80]
    percentage = over80.shape[0] / acc.shape[0]
    experiment_logger.experiment.add_scalar(f'{model_name}/80/test', percentage, global_step=global_step)
