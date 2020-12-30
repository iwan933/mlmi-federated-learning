import random
from typing import List

from mlmi.participant import BaseParticipant


class BaseSelector(object):

    def select_participants(self, participants: List['BaseParticipant']):
        raise NotImplementedError()


class RandomParticipantFractionSelector(BaseSelector):

    def __init__(self, fraction: float = 1.0):
        self._fraction = fraction

    def select_participants(self, participants: List['BaseParticipant']):
        target_num = int(self._fraction * len(participants)) + 1
        indices = [random.randint()]
