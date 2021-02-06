# Reptile in the federated learning setting

Reptile is a meta-learning algorithm. This adaptation applies it to the federated learning context.

This implementation is based on the [code](https://github.com/openai/supervised-reptile) from the [Nichol 2018 paper](https://arxiv.org/abs/1803.02999). Only the [Omniglot](https://github.com/brendenlake/omniglot) dataset is used.

# Available experiments

There are three runnable experiments:
* `run_reptile_original.py` runs the original experimental setting from the Nichol paper, including the same data loading. Minor but important adaptations have been made: (1) command-line options not relevant to the federated learning setting have been removed and (2) the optimizer state is reset after every task. The latter was not the case in the original source.
* `run_reptile_original_dataloading_federated.py` runs an experiment with the TensorFlow model from the Nichol paper, but with a federated data loading design of the Omniglot data.
* `run_reptile_federated.py` runs the full adaptation of Reptile to the federated learning setting

All experiments can be specified with the same command-line options:
```
--seed:             random seed (default 0)
--classes:          number of classes per inner task (default 5)
--shots:            number of examples per class (default 5)
--inner-batch:      inner batch size (default 10)
--inner-iters:      inner iterations (default 5)
--learning-rate:    Adam step size (default 1e-3)
--meta-step:        meta-training step size (default 1)
--meta-step-final:  meta-training step size by the end (default 0)
--meta-batch:       meta-training batch size (default 5)
--meta-iters:       meta-training iterations (default 3000)
--eval-iters:       eval inner iterations (default 50)
--eval-samples:     evaluation samples (default 100)
--eval-interval:    train steps per eval (default 10)
--sgd:              use gradient descent instead of Adam as inner optimizer
```
In addition, `run_reptile_original_dataloading_federated.py` and `run_reptile_federated.py` take the following options:
```
--train-clients:    number of training clients (default 10000)
--test-clients:     number of test clients (default 1000)
```

# Original specifications of the Omniglot experiments in the Nichol paper

```shell
# transductive 1-shot 5-way Omniglot.
python -u run_omniglot.py --shots 1 --inner-batch 10 --inner-iters 5 --meta-step 1 --meta-batch 5 --meta-iters 100000 --eval-batch 5 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots 10 --checkpoint ckpt_o15t --transductive

# 5-shot 5-way Omniglot.
python -u run_omniglot.py --train-shots 10 --inner-batch 10 --inner-iters 5 --learning-rate 0.001 --meta-step 1 --meta-step-final 0 --meta-batch 5 --meta-iters 100000 --eval-batch 5 --eval-iters 50 --checkpoint ckpt_o55

# 1-shot 5-way Omniglot.
python -u run_omniglot.py --shots 1 --inner-batch 10 --inner-iters 5 --meta-step 1 --meta-batch 5 --meta-iters 100000 --eval-batch 5 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots 10 --checkpoint ckpt_o15

# 1-shot 20-way Omniglot.
python -u run_omniglot.py --shots 1 --classes 20 --inner-batch 20 --inner-iters 10 --meta-step 1 --meta-batch 5 --meta-iters 200000 --eval-batch 10 --eval-iters 50 --learning-rate 0.0005 --meta-step-final 0 --train-shots 10 --checkpoint ckpt_o120

# 5-shot 20-way Omniglot.
python -u run_omniglot.py --classes 20 --inner-batch 20 --inner-iters 10 --meta-step 1 --meta-batch 5 --meta-iters 200000 --eval-batch 10 --eval-iters 50 --learning-rate 0.0005 --meta-step-final 0 --train-shots 10 --checkpoint ckpt_o520
```