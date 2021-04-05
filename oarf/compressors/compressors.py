import random
import logging
from oarf.utils.module import Module
from oarf.compressors.none import NoCompress
from oarf.compressors.sparse import SparseCompress
from oarf.compressors.topk import TopKPerLayer, TopKPerModel
from oarf.compressors.randk import RandKPerLayer, RandKPerModel
from oarf.compressors.rankk import RankKPerLayer
from oarf.compressors.svd import SVDPerLayer
from oarf.compressors.adarankk import AdaRankKPerLayer
from oarf.compressors.base import (
    _SubsampleCompressor, _RandomCompressor, _LowRankCompressor)

log = logging.getLogger(__name__)

compressor_name_map = {
    'none': NoCompress,
    'sparse': SparseCompress,
    'topk_perlayer': TopKPerLayer,
    'topk_permodel': TopKPerModel,
    'randk_perlayer': RandKPerLayer,
    'randk_permodel': RandKPerModel,
    'rankk_perlayer': RankKPerLayer,
    'svd_perlayer': SVDPerLayer,
    'ada_rankk_perlayer': AdaRankKPerLayer
}


class Compressor(Module):
    """Any class that needs compressors can inherit this class"""

    def __init__(
            self, clientside_compressor, serverside_compressor,
            seed=None, client_ratio: float = None, server_ratio: float = None,
            client_rank: int = None, server_rank: int = None, **kwargs):
        super().__init__(**kwargs, seed=seed)
        self.c_compressor = self.init_compressor(
            clientside_compressor, seed, client_ratio, client_rank)
        self.s_compressor = self.init_compressor(
            serverside_compressor, seed, server_ratio, server_rank)

    @staticmethod
    def init_compressor(
            name: str, seed=None, ratio: float = None, rank: int = None):

        # first initialize compressor with default arguments
        compressor = compressor_name_map[name]()
        log.debug("Using compressor: {}".format(type(compressor)))

        # set ratio for client/server compressor, same ratio for all levels
        if isinstance(compressor, _SubsampleCompressor):
            assert ratio is not None, \
                "ratio must be specified when using a subsample compressor."
            compressor.set_ratios(ratio)

        # set ranks for client/server compressor, same rank for all levels
        if isinstance(compressor, _LowRankCompressor):
            assert rank is not None, \
                "rank must be specified when using a lowrank compressor."
            compressor.set_ranks(rank)

        if isinstance(compressor, _RandomCompressor):
            assert seed is not None, \
                "seed must be specified when using a random compressor."
            random.seed(str(seed) + __name__)
            compressor.set_seed(random.random())

        return compressor
