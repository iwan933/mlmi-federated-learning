

class HierarchicalExperimentContext(object):

    def __init__(self):
        super().__init__()
        self._cluster_args = None

    @property
    def cluster_args(self) -> 'ClusterArgs':
        return self._cluster_args

    @cluster_args.setter
    def cluster_args(self, value: 'ClusterArgs'):
        self._cluster_args = value
