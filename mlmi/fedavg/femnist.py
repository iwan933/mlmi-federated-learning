from fedml_api.data_preprocessing.FederatedEMNIST.data_loader import load_partition_data_federated_emnist

from mlmi.struct import FederatedDatasetData


def load_femnist_dataset(data_dir, num_clients=3400, batch_size=20) -> FederatedDatasetData:
    """
    Load the federated tensorflow emnist dataset, originally split up into 3400 clients.
    :param data_dir: data directory
    :param num_clients: number of clients to use for split
    :return:
    """
    federated_dataset_args = load_partition_data_federated_emnist('', data_dir, client_number=num_clients,
                                                                  batch_size=batch_size)
    return FederatedDatasetData(*federated_dataset_args)
