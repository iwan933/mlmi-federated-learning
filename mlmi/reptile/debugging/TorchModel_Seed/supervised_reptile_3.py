"""
Supervised Reptile learning and evaluation on arbitrary
datasets.
"""

import copy
import random
import torch
import pytorch_lightning as pl
from itertools import cycle

from mlmi.reptile.model import OmniglotLightning, weight_model, sum_model_states, subtract_model_states
from .variables_3 import (interpolate_vars, average_vars, subtract_vars, add_vars, scale_vars,
                          VariableState)

class Reptile:
    """
    A meta-learning session.

    Reptile can operate in two evaluation modes: normal
    and transductive. In transductive mode, information is
    allowed to leak between test samples via BatchNorm.
    Typically, MAML is used in a transductive manner.
    """
    def __init__(self, global_model, model_kwargs, inner_iterations, inner_iterations_eval):
        self.model_kwargs = model_kwargs
        self.criterion = torch.nn.CrossEntropyLoss()
        self.inner_iterations = inner_iterations
        self.inner_iterations_eval = inner_iterations_eval
        self.global_model = global_model



    # pylint: disable=R0913,R0914
    def train_step(self,
                   meta_batch,
                   meta_step_size):
        """
        Perform a Reptile training step.

        Args:
          dataset: a sequence of data classes, where each data
            class has a sample(n) method.
          input_ph: placeholder for a batch of samples.
          label_ph: placeholder for a batch of labels.
          minimize_op: TensorFlow Op to minimize a loss on the
            batch specified by input_ph and label_ph.
          num_classes: number of data classes to sample.
          num_shots: number of examples per data class.
          inner_batch_size: batch size for every inner-loop
            training iteration.
          inner_iters: number of inner-loop iterations.
          replacement: sample with replacement.
          meta_step_size: interpolation coefficient.
          meta_batch_size: how many inner-loops to run.
        """
        old_vars = copy.deepcopy(self.global_model.state_dict())
        new_vars = []
        for key, data_loader in meta_batch.items():
            model = OmniglotLightning(participant_name='local_model', **self.model_kwargs)
            model.load_state_dict(copy.deepcopy(old_vars))
            trainer = pl.Trainer(
                checkpoint_callback=False,
                limit_val_batches=0.0,
                min_steps=self.inner_iterations,
                max_steps=self.inner_iterations,
                weights_summary=None,
                progress_bar_refresh_rate=0
            )
            trainer.fit(model, data_loader, data_loader)
            new_vars.append(copy.deepcopy(model.state_dict()))
            #new_vars.append(weight_model(copy.deepcopy(self.model.state_dict()), 1, len(meta_batch)))

        new_vars = average_vars(new_vars)  #sum_model_states(new_vars)
        #meta_gradient = subtract_model_states(new_vars, old_vars)
        #self.update_model_state(old_vars, meta_gradient, meta_step_size)
        self.global_model.load_state_dict(interpolate_vars(old_vars, new_vars, meta_step_size))

    def update_model_state(self, old_vars, gradient, learning_rate):
        """
        Update model state with vanilla gradient descent
        :param gradient: OrderedDict[str, Tensor]
        :return:
        """
        # TODO (optional): Extend this function with other optimizer options
        #                  than vanilla GD
        new_model_state = copy.deepcopy(old_vars)
        for key, w in new_model_state.items():
            if key.endswith('running_mean') or key.endswith('running_var') \
                or key.endswith('num_batches_tracked'):
                # Do not update non-trainable batch norm parameters
                continue
            new_model_state[key] = w + learning_rate * gradient[key]
        self.model.load_state_dict(new_model_state)

    def evaluate(self,
                 train_data_loader,
                 test_data_loader):
        """
        Run a single evaluation of the model.

        Samples a few-shot learning task and measures
        performance.

        Args:
          dataset: a sequence of data classes, where each data
            class has a sample(n) method.
          input_ph: placeholder for a batch of samples.
          label_ph: placeholder for a batch of labels.
          minimize_op: TensorFlow Op to minimize a loss on the
            batch specified by input_ph and label_ph.
          predictions: a Tensor of integer label predictions.
          num_classes: number of data classes to sample.
          num_shots: number of examples per data class.
          inner_batch_size: batch size for every inner-loop
            training iteration.
          inner_iters: number of inner-loop iterations.
          replacement: sample with replacement.

        Returns:
          The number of correctly predicted samples.
            This always ranges from 0 to num_classes.
        """

        old_vars = copy.deepcopy(self.global_model.state_dict())

        model = OmniglotLightning(participant_name='local_model', **self.model_kwargs)
        model.load_state_dict(copy.deepcopy(old_vars))
        trainer = pl.Trainer(
            checkpoint_callback=False,
            limit_val_batches=0.0,
            min_steps=self.inner_iterations_eval,
            max_steps=self.inner_iterations_eval,
            weights_summary=None,
            progress_bar_refresh_rate=0
        )
        trainer.fit(model, train_data_loader, train_data_loader)

        #for i, (inputs, labels) in zip(range(self.inner_iterations_eval), cycle(train_data_loader)):
        #    self.optimizer.zero_grad()
        #    outputs = self.model(inputs)
        #    loss = self.criterion(outputs, labels)
        #    loss.backward()
        #    self.optimizer.step()
        inputs, labels = list(test_data_loader)[0]
        test_preds = model(inputs).argmax(dim=1)
        num_correct = int(sum([pred == label for pred, label in zip(test_preds, labels)]))
        return num_correct
