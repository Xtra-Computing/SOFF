from oarf.algorithms.fedavg import FedAvgServer, FedAvgClient


# fedsgd is a special case of fedavg
class FedSGDServer(FedAvgServer):
    def __init__(self, **kwargs):
        kwargs['client_fraction'] = 1
        super().__init__(**kwargs)


FedSGDClient = FedAvgClient

__all__ = ['FedSGDServer', 'FedSGDClient']
