from typing import Dict, List
from collections import OrderedDict

from torch import Tensor
import pytorch_lightning as pl

from mlmi.log import getLogger
from mlmi.participant import BaseParticipantModel, BaseTrainingParticipant, BaseAggregatorParticipant, BaseParticipant
from mlmi.struct import TrainArgs, ModelArgs, ExperimentContext


logger = getLogger(__name__)

def weight_model(model: Dict[str, Tensor], num_samples: int, num_total_samples: int) -> Dict[str, Tensor]:
    weighted_model_state = OrderedDict()
    for key, w in model.items():
        weighted_model_state[key] = (num_samples / num_total_samples) * w
    return weighted_model_state

def sum_model_states(model_state_list: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    result_state = model_state_list[0].copy()
    for model_state in model_state_list[1:]:
        for key, w in model_state.items():
            result_state[key] += w
    return result_state

def subtract_model_states(minuend: OrderedDict,
                          subtrahend: OrderedDict) -> OrderedDict:
    """
    Returns difference of two model_states: minuend - subtrahend
    """
    result_state = minuend.copy()
    for key, w in subtrahend.items():
        result_state[key] -= w
    return result_state

class ReptileClient(BaseTrainingParticipant):

    def test(self, training_args: TrainArgs):
        """
        Test the model state on this client's data.
        :param
        :param model_state: The model state to evaluate
        :return: The output loss
        """

        trainer = self.create_trainer(enable_logging=False,
                                      **training_args.kwargs)
        train_dataloader = self.test_data_loader[0]
        trainer.fit(self.model, train_dataloader, train_dataloader)
        self.save_model_state()

        train_set, test_set = self.test_data_loader[1], self.train_data_loader[2]
        test_preds = self._test_predictions(train_set=train_set, test_set=test_set)
        # TODO: Compute loss and accuracy
        #num_correct = sum([pred == sample[1] for pred, sample in zip(test_preds, test_set)])
        #self._full_state.import_variables(old_vars)
        #return num_correct

        #result = trainer.test(model=model, test_dataloaders=self.test_data_loader)
        #return result

    def _test_predictions(self, train_set, test_set):
        res = []
        for test_sample in test_set:
            inputs, _ = zip(*train_set)
            inputs += (test_sample[0],)
            res.append(self.model(inputs)[-1])
        return res

class ReptileServer(BaseAggregatorParticipant):
    def __init__(self,
                 participant_name: str,
                 context: ExperimentContext,
                 initial_model_state: OrderedDict = None):
        super().__init__(participant_name, None, context)
        # Initialize model parameters
        if initial_model_state is not None:
            self.model.load_state_dict(initial_model_state)

    @property
    def model_args(self):
        return self._model_args

    def aggregate(self,
                  participants: List[BaseParticipant],
                  meta_learning_rate: float,
                  weighted: bool = True):

        # Collect participants' model states and calculate model differences to
        # initial model (= model deltas)
        initial_model_state = self.model.state_dict()
        participant_model_deltas = []
        for participant in participants:
            participant_model_deltas.append(
                subtract_model_states(
                    participant.model.state_dict(), initial_model_state
                )
            )
        if weighted:
            # meta_gradient = weighted (by number of samples) average of
            # participants' model updates
            num_train_samples = []
            for participant in participants:
                num_train_samples.append(participant.num_train_samples)
            weighted_model_delta_list = []
            num_total_samples = sum(num_train_samples)
            for num_samples, pmd in zip(num_train_samples, participant_model_deltas):
                weighted_model_delta = weight_model(
                    pmd, num_samples, num_total_samples
                )
                weighted_model_delta_list.append(weighted_model_delta)
            meta_gradient = sum_model_states(weighted_model_delta_list)
            self.total_train_sample_num = num_total_samples
        else:
            # meta_gradient = simple average of participants' model updates
            scaled_model_delta_list = []
            for pmd in participant_model_deltas:
                scaled_model_delta = weight_model(
                    pmd, 1, len(participant_model_deltas)
                )
                scaled_model_delta_list.append(scaled_model_delta)
            meta_gradient = sum_model_states(scaled_model_delta_list)

        # Update model state with meta_gradient using simple gradient descent
        self.update_model_state(meta_gradient, meta_learning_rate)

    def update_model_state(self, gradient, learning_rate):
        """
        Update model state using gradient
        :param gradient: OrderedDict[str, Tensor]
        :return:
        """
        # TODO (optional): Extend this function with other optimizer options
        #                  than vanilla GD
        new_model_state = self.model.state_dict().copy()
        for key, w in new_model_state.items():
            new_model_state[key] = w + \
                learning_rate * gradient[key]
        self.model.load_state_dict(new_model_state)

class OmniglotModel(BaseParticipantModel, pl.LightningModule):
    # TODO: Implement model as below in PyTorch (this is the classifier from
    #       Nichols 2018: On First-Order Meta-Learning Algorithms

class OmniglotModel:
    """
    A model for Omniglot classification.
    """

    def __init__(self, num_classes, optimizer=DEFAULT_OPTIMIZER, **optim_kwargs):
        self.input_ph = tf.placeholder(tf.float32, shape=(None, 28, 28))
        out = tf.reshape(self.input_ph, (-1, 28, 28, 1))
        for _ in range(4):
            out = tf.layers.conv2d(out, 64, 3, strides=2, padding='same')
            out = tf.layers.batch_normalization(out, training=True)
            out = tf.nn.relu(out)
        out = tf.reshape(out, (-1, int(np.prod(out.get_shape()[1:]))))
        self.logits = tf.layers.dense(out, num_classes)
        self.label_ph = tf.placeholder(tf.int32, shape=(None,))
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_ph,
                                                                   logits=self.logits)
        self.predictions = tf.argmax(self.logits, axis=-1)
        self.minimize_op = optimizer(**optim_kwargs).minimize(self.loss)
