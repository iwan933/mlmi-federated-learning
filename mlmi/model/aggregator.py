from typing import List, Dict
from torch import nn, Tensor


class BaseAggregator(object):

    def aggregate_models(self, model_to_update: Dict[str, Tensor], models_to_aggregate: List[Dict[str, Tensor]],
                         *args, **kwargs) -> Dict[str, Tensor]:
        """
        Aggregates models of a communication round
        :param model_to_update: The model that should incoporate the changes
        :param models_to_aggregate: The various models that should be aggregated
        :return:
        """
        raise NotImplementedError()
