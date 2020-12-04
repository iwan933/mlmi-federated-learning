import numpy as np
import random
import wandb

import torch
from fedml_experiments.standalone.fedavg.main_fedavg import add_args as _add_args, create_model, load_data
from fedml_api.standalone.fedavg.fedavg_trainer import FedAvgTrainer

from mlmi.log import getLogger


add_args = _add_args
logger = getLogger(__name__)


def execute(args):
    logger.info(args)
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    logger.info(device)

    wandb.init(
        project="fedml",
        name="FedAVG-r" + str(args.comm_round) + "-e" + str(args.epochs) + "-lr" + str(args.lr),
        config=args
    )

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # load data
    dataset = load_data(args, args.dataset)

    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)
    model = create_model(args, model_name=args.model, output_dim=dataset[7])
    logger.info(model)

    trainer = FedAvgTrainer(dataset, model, device, args)
    trainer.train()
