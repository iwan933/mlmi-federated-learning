"""
Command-line argument parsing.
"""

import argparse


def argument_parser():
    """
    Get an argument parser for a training script.
    """
    # Changes from original code: default values adapted to our needs
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='random seed', default=0, type=int)
    parser.add_argument('--train-clients', help='number of training clients', default=10000,
                        type=int)
    parser.add_argument('--test-clients', help='number of test clients', default=1000, type=int)
    parser.add_argument('--classes', help='number of classes per inner task', default=5, type=int)
    parser.add_argument('--shots', help='number of examples per class', default=5, type=int)
    parser.add_argument('--inner-batch', help='inner batch size', default=10, type=int)
    parser.add_argument('--inner-iters', help='inner iterations', default=5, type=int)
    parser.add_argument('--learning-rate', help='Adam step size', default=1e-3, type=float)
    parser.add_argument('--meta-step', help='meta-training step size', default=1, type=float)
    parser.add_argument('--meta-step-final', help='meta-training step size by the end',
                        default=0, type=float)
    parser.add_argument('--meta-batch', help='meta-training batch size', default=5, type=int)
    parser.add_argument('--meta-iters', help='meta-training iterations', default=3000, type=int)
    parser.add_argument('--eval-iters', help='eval inner iterations', default=50, type=int)
    parser.add_argument('--eval-samples', help='evaluation samples', default=100, type=int)
    parser.add_argument('--eval-interval', help='train steps per eval', default=10, type=int)
    parser.add_argument('--sgd', help='use vanilla SGD instead of Adam', action='store_true')
    return parser
