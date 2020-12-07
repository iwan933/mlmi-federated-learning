import os
import argparse

from mlmi.log import getLogger
from mlmi.fedml_reference import add_args, execute


logger = getLogger(__name__)


if __name__ == '__main__':
    parser = add_args(argparse.ArgumentParser(description='FedAvg-standalone'))
    args = parser.parse_args()

    wandb_api_key = os.getenv('WANDB_API_KEY')
    logger.info('running with wandb api key: {0}*****'.format(wandb_api_key[0:5]))
    execute(args)
