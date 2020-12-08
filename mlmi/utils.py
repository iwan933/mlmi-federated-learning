from pytorch_lightning.loggers import TensorBoardLogger
from mlmi.settings import RUN_DIR


def create_tensorboard_logger(experiment_name: str, client_name: str) -> TensorBoardLogger:
    experiment_path = RUN_DIR / experiment_name / client_name
    return TensorBoardLogger(experiment_path.absolute())
