"""
Tools for manipulating sets of variables.
"""

import numpy as np
import tensorflow.compat.v1 as tf
from collections import OrderedDict
import torch

def interpolate_vars(old_vars, new_vars, epsilon):
    """
    Interpolate between two sequences of variables.
    """
    #return add_vars(old_vars, scale_vars(subtract_vars(new_vars, old_vars), epsilon))
    res = OrderedDict()
    for key, item in old_vars.items():
        if key.endswith('running_mean') or key.endswith('num_batches_tracked'):
            res[key] = torch.zeros_like(item)
            continue
        if key.endswith('running_var'):
            res[key] = torch.ones_like(item)
            continue
        res[key] = old_vars[key] + (new_vars[key] - old_vars[key]) * epsilon
    return res

def average_vars(var_seqs):
    """
    Average a sequence of variable sequences.
    """
    #res = []
    #for variables in zip(*var_seqs):
    #    res.append(np.mean(variables, axis=0))
    #return res
    res = OrderedDict()
    for key, item in var_seqs[0].items():
        if key.endswith('running_mean') or key.endswith('num_batches_tracked'):
            res[key] = torch.zeros_like(item)
            continue
        if key.endswith('running_var'):
            res[key] = torch.ones_like(item)
            continue
        res[key] = torch.zeros_like(item)
        for i in range(len(var_seqs)):
            res[key] += var_seqs[i][key] / len(var_seqs)
    return res


def subtract_vars(var_seq_1, var_seq_2):
    """
    Subtract one variable sequence from another.
    """
    return [v1 - v2 for v1, v2 in zip(var_seq_1, var_seq_2)]

def add_vars(var_seq_1, var_seq_2):
    """
    Add two variable sequences.
    """
    return [v1 + v2 for v1, v2 in zip(var_seq_1, var_seq_2)]

def scale_vars(var_seq, scale):
    """
    Scale a variable sequence.
    """
    return [v * scale for v in var_seq]

def weight_decay(rate, variables=None):
    """
    Create an Op that performs weight decay.
    """
    if variables is None:
        variables = tf.trainable_variables()
    ops = [tf.assign(var, var * rate) for var in variables]
    return tf.group(*ops)

class VariableState:
    """
    Manage the state of a set of variables.
    """
    def __init__(self, session, variables):
        self._session = session
        self._variables = variables
        self._placeholders = [tf.placeholder(v.dtype.base_dtype, shape=v.get_shape())
                              for v in variables]
        assigns = [tf.assign(v, p) for v, p in zip(self._variables, self._placeholders)]
        self._assign_op = tf.group(*assigns)

    def export_variables(self):
        """
        Save the current variables.
        """
        return self._session.run(self._variables)

    def import_variables(self, values):
        """
        Restore the variables.
        """
        self._session.run(self._assign_op, feed_dict=dict(zip(self._placeholders, values)))
