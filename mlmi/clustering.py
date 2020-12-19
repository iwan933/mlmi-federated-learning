from typing import Dict, List
import random

from mlmi.participant import BaseParticipant


class BaseClusterPartitioner(object):

    def cluster(self, participants: List[BaseParticipant]) -> Dict[str, List[BaseParticipant]]:
        raise NotImplementedError()


class RandomClusterPartitioner(BaseClusterPartitioner):

    def cluster(self, participants: List[BaseParticipant]) -> Dict[str, List[BaseParticipant]]:
        num_cluster = 10
        result_dic = {}
        for id in range(1, num_cluster+1):
            result_dic[str(id)] = []
        for participant in participants:
            participant.cluster_id = str(random.randint(1, num_cluster))
            result_dic[participant.cluster_id].append(participant)
        return result_dic


