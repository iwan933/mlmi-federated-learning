from torch.utils.tensorboard import SummaryWriter
from mlmi.settings import RUN_DIR


def create_tensorboard_writer(experiment_name: str) -> SummaryWriter:
    experiment_path = RUN_DIR / experiment_name
    return SummaryWriter(experiment_path.absolute())
