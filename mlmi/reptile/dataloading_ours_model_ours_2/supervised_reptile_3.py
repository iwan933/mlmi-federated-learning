"""
Supervised Reptile learning and evaluation on arbitrary
datasets.
"""

import copy
import random
import torch
from itertools import cycle

from mlmi.reptile.model import weight_model, sum_model_states, subtract_model_states
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
    def __init__(self, model, optimizer, inner_iterations, inner_iterations_eval):
        self.model = model
        self.optimizer = optimizer
        self.criterion = torch.nn.CrossEntropyLoss()
        self.inner_iterations = inner_iterations
        self.inner_iterations_eval = inner_iterations_eval

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
        old_vars = copy.deepcopy(self.model.state_dict())
        old_vars_optim = copy.deepcopy(self.optimizer.state_dict())
        new_vars = []
        for key, data_loader in meta_batch.items():
            for i, (inputs, labels) in zip(range(self.inner_iterations), cycle(data_loader)):
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            new_vars.append(copy.deepcopy(self.model.state_dict()))
            #new_vars.append(weight_model(copy.deepcopy(self.model.state_dict()), 1, len(meta_batch)))
            self.model.load_state_dict(old_vars)
            self.optimizer.load_state_dict(old_vars_optim)

        new_vars = average_vars(new_vars)  #sum_model_states(new_vars)
        #meta_gradient = subtract_model_states(new_vars, old_vars)
        #self.update_model_state(old_vars, meta_gradient, meta_step_size)
        self.model.load_state_dict(interpolate_vars(old_vars, new_vars, meta_step_size))

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

        old_vars = copy.deepcopy(self.model.state_dict())
        old_vars_optim = copy.deepcopy(self.optimizer.state_dict())
        for i, (inputs, labels) in zip(range(self.inner_iterations_eval), cycle(train_data_loader)):
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
        inputs, labels = list(test_data_loader)[0]
        test_preds = self.model(inputs).argmax(dim=1)
        num_correct = int(sum([pred == label for pred, label in zip(test_preds, labels)]))
        self.model.load_state_dict(old_vars)
        self.optimizer.load_state_dict(old_vars_optim)
        return num_correct


def _sample_mini_dataset(dataset, num_classes, num_shots):
    """
    Sample a few shot task from a dataset.

    Returns:
      An iterable of (input, label) pairs.
    """
    shuffled = list(dataset)
    random.shuffle(shuffled)
    for class_idx, class_obj in enumerate(shuffled[:num_classes]):
        for sample in class_obj.sample(num_shots):
            yield (sample, class_idx)

def _mini_batches(samples, batch_size, num_batches, replacement):
    """
    Generate mini-batches from some data.

    Returns:
      An iterable of sequences of (input, label) pairs,
        where each sequence is a mini-batch.
    """
    samples = list(samples)
    if replacement:
        for _ in range(num_batches):
            yield random.sample(samples, batch_size)
        return
    cur_batch = []
    batch_count = 0
    while True:
        random.shuffle(samples)
        for sample in samples:
            cur_batch.append(sample)
            if len(cur_batch) < batch_size:
                continue
            yield cur_batch
            cur_batch = []
            batch_count += 1
            if batch_count == num_batches:
                return

def _split_train_test(samples, test_shots=1):
    """
    Split a few-shot task into a train and a test set.

    Args:
      samples: an iterable of (input, label) pairs.
      test_shots: the number of examples per class in the
        test set.

    Returns:
      A tuple (train, test), where train and test are
        sequences of (input, label) pairs.
    """
    train_set = list(samples)
    test_set = []
    labels = set(item[1] for item in train_set)
    for _ in range(test_shots):
        for label in labels:
            for i, item in enumerate(train_set):
                if item[1] == label:
                    del train_set[i]
                    test_set.append(item)
                    break
    if len(test_set) < len(labels) * test_shots:
        raise IndexError('not enough examples of each class for test set')
    return train_set, test_set
