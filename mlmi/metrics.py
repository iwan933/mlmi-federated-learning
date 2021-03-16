from typing import Any, Callable, Optional

import torch
import numpy as np
from pytorch_lightning.metrics import Metric
from pytorch_lightning.metrics.utils import _input_format_classification
from sklearn.metrics import balanced_accuracy_score


class BalancedAccuracy(Metric):
    def __init__(
        self,
        threshold=0.5,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state("tracked_preds", default=torch.Tensor([]), dist_reduce_fx="sum")
        self.add_state("tracked_targets", default=torch.Tensor([]), dist_reduce_fx="sum")

        self.threshold = threshold

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        preds, target = _input_format_classification(preds, target, self.threshold)
        assert preds.shape == target.shape

        self.tracked_preds = torch.cat((self.tracked_preds, preds))
        self.tracked_targets = torch.cat((self.tracked_targets, target))

    def compute(self):
        """
        Computes accuracy over state.
        """
        targets = self.tracked_targets.cpu().numpy()
        preds = self.tracked_preds.cpu().numpy()
        if len(targets) == 0:
            return 0.0
        result = balanced_accuracy_score(targets, preds)
        result = torch.FloatTensor([result])
        if torch.cuda.is_available():
            return result.cuda().float()
        return result.float()
