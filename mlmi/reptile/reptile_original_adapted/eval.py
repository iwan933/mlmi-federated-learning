"""
Helpers for evaluating models.
"""

from .reptile import ReptileForFederatedData
from mlmi.reptile.reptile_original.variables import weight_decay

# pylint: disable=R0913,R0914
def evaluate(sess,
             model,
             train_dataloaders,
             test_dataloaders,
             num_classes=5,
             eval_inner_iters=50,
             transductive=False,
             weight_decay_rate=1,
             reptile_fn=ReptileForFederatedData):
    """
    Evaluate a model on a dataset.
    """
    reptile = reptile_fn(
        session=sess,
        transductive=transductive,
        pre_step_op=weight_decay(weight_decay_rate)
    )
    total_correct = 0
    for train_dl, test_dl in zip(
            train_dataloaders.values(), test_dataloaders.values()
    ):
        total_correct += reptile.evaluate(
            train_data_loader=train_dl,
            test_data_loader=test_dl,
            input_ph=model.input_ph,
            label_ph=model.label_ph,
            minimize_op=model.minimize_op,
            predictions=model.predictions,
            inner_iters=eval_inner_iters
        )
    return total_correct / (len(train_dataloaders) * num_classes)
