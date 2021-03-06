import numpy as np
from typing import List

from mlmi.participant import BaseTrainingParticipant


def sample_randomly_by_fraction(participants: List['BaseTrainingParticipant'],
                                fraction: float) -> List['BaseTrainingParticipant']:
    assert 0.0 <= fraction <= 1.0, 'Client fraction (cf) has to fulfill 0.0 <= cf <= 1.0'

    out = []
    num_participants = len(participants)
    num_out = min(max(int(1.3 * fraction * num_participants), 1), num_participants)
    indices = np.arange(num_participants)
    random_indices = np.random.choice(indices, size=num_out, replace=False)
    for i in random_indices:
        out.append(participants[i])
    return out
