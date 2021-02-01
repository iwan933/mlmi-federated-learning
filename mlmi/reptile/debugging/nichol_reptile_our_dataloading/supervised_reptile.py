"""
Supervised Reptile learning and evaluation on arbitrary
datasets.
"""

import tensorflow.compat.v1 as tf
from itertools import cycle

from mlmi.reptile.reptile_original.variables import (
    interpolate_vars, average_vars, VariableState
)


class ReptileForDataloaders:
    """
    A meta-learning session.

    Reptile can operate in two evaluation modes: normal
    and transductive. In transductive mode, information is
    allowed to leak between test samples via BatchNorm.
    Typically, MAML is used in a transductive manner.
    """
    def __init__(self, session, variables=None, transductive=False, pre_step_op=None):
        self.session = session
        self._model_state = VariableState(self.session, variables or tf.trainable_variables())
        self._full_state = VariableState(self.session,
                                         tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        self._transductive = transductive
        self._pre_step_op = pre_step_op

    # pylint: disable=R0913,R0914
    def train_step(self,
                   meta_batch,
                   input_ph,
                   label_ph,
                   minimize_op,
                   inner_iters,
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
        # Change from original code: Save and reset _full_state to reset not
        # only model state but also optimizer state. The original implementation
        # transmits optimizer parameters between tasks which leads to improved
        # performance but is not applicable to the federated learning setting.
        full_old_vars = self._full_state.export_variables()
        old_vars = self._model_state.export_variables()
        new_vars = []
        # Change from original code: Use provided training data_loader for data
        # retrieval rather than generating sampling training data on the spot in
        # the original.
        for key, data_loader in meta_batch.items():
            for i, batch in zip(range(inner_iters), cycle(data_loader)):
                inputs = []
                for t in batch[0]:
                    inputs.append(t.numpy())
                inputs = tuple(inputs)
                labels = tuple(batch[1].numpy())
                self.session.run(
                    minimize_op, feed_dict={input_ph: inputs, label_ph: labels}
                )
            new_vars.append(self._model_state.export_variables())
            ####
            # Change from original code: load full state and not only model state
            self._full_state.import_variables(full_old_vars)
            # self._model_state.import_variables(old_vars) <- This was the original code
            ####
        new_vars = average_vars(new_vars)

        self._model_state.import_variables(interpolate_vars(old_vars, new_vars, meta_step_size))


    def evaluate(self,
                 train_data_loader,
                 test_data_loader,
                 input_ph,
                 label_ph,
                 minimize_op,
                 predictions,
                 inner_iters):
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

        old_vars = self._full_state.export_variables()

        # Change from original code: Use specified training and test data_loaders
        # for data retrieval rather than generating data on the spot in the original
        for i, batch in zip(range(inner_iters), cycle(train_data_loader)):
            inputs = []
            for t in batch[0]:
                inputs.append(t.numpy())
            inputs = tuple(inputs)
            labels = tuple(batch[1].numpy())
            if self._pre_step_op:
                self.session.run(self._pre_step_op)
            self.session.run(minimize_op, feed_dict={input_ph: inputs, label_ph: labels})

        _test_set = list(test_data_loader)[0]
        inputs = []
        for t in _test_set[0]:
            inputs.append(t.numpy())
        labels = list(_test_set[1].numpy())
        test_set = list(zip(inputs, labels))

        test_preds = self._test_predictions(test_set, input_ph, predictions)
        num_correct = sum([pred == sample[1] for pred, sample in zip(test_preds, test_set)])
        self._full_state.import_variables(old_vars)
        return num_correct

    def _test_predictions(self, test_set, input_ph, predictions):
        inputs, _ = zip(*test_set)
        return self.session.run(predictions, feed_dict={input_ph: inputs})
