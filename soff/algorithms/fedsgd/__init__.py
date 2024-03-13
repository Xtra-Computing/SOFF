"""The FedSGD algorithms"""
from ..base.base_server import BaseServer
from .. import fedavg

# fedsgd is a special case of fedavg


class _FedSGDAdaptor(BaseServer):
    """Adaptor to turn fedavg server into a FedSGD server"""

    def __init__(self, cfg):
        cfg.fedavg.client_fraction = 1.0
        super().__init__(cfg)


class FedSGDServer(_FedSGDAdaptor, fedavg.Server):
    """Simple FedSGD server"""


class Client(fedavg.Client):
    """Simple FedSGD client"""
