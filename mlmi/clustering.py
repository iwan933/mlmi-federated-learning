from typing import Dict, List

from mlmi.participant import BaseParticipant


class BaseClusterPartitioner(object):

    def cluster(self, participants: List[BaseParticipant]) -> Dict[str, List[BaseParticipant]]:
        raise NotImplementedError()


class RandomClusterPartitioner(BaseClusterPartitioner):

    def cluster(self, participants: List[BaseParticipant]) -> Dict[str, List[BaseParticipant]]:
        raise NotImplementedError()
