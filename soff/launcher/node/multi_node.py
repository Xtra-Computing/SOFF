"""Simulates multiple nodes in one main thread"""

import os
import sys
import signal
import threading
from threading import Thread, Semaphore, Lock, get_ident
from typing import List, Sequence
import torch
from munch import Munch
from ...algorithms.base.base_client import ResourceManagedClient
from .node import NodeLauncher, BaseLauncherConfParser, alg_name_map


class MultiNodeConfParser(BaseLauncherConfParser):
    """Parse config for multi-node launcher"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        multicli_args = self.add_argument_group(
            "Multi-Node Launcher Configs (C)",
            description="  Use -- to separate arguments for launcher and "
            "arguments for nodes and also arguments for each node. E.g. "
            "`-l.a fedavg -l.mc.n 2 -- <args-for-cli-0> -- <args-for-cli-1>`")

        multicli_args.add_argument(
            '-l.mn.n', '--launcher.multi-node.num-nodes',
            default=1, type=int, metavar='N',
            help="Number of nodes class to instantiate. This is not the total "
            "number of nodes, but the number of nodes instantiated by the "
            "multi-node simulator.")

        multicli_args.add_argument(
            '-l.mn.mnpg', '--launcher.multi-node.max-nodes-per-gpu',
            default=[65535], type=int, nargs='+', metavar='M',
            help="max number of nodes that can run simultaneously on each "
            "GPU. The length of this list must be 1 or the same as the number "
            "of gpu devices. If 1, the limit applies to all GPUs.")


class MultiNodeLauncher(NodeLauncher):
    """
    A MultiNodeLauncher objects simulates multiple nodes in one main
    process, with each node a separate thread, and manages the GPU resources
    allocation among the nodes. While we can directly use multiple node
    processes and schedule the resource allocation with IPC, each node
    will then initialize a separate cuDNN context, which takes around 1GiB,
    which can cause a huge problem when the number of nodes are large and
    number of GPUs are limited. The only way to make the nodes share same
    cuDNN contexts are to use a single process to initialize multiple nodes.
    """
    @classmethod
    def start(cls, launcher_conf_parser_class=MultiNodeConfParser):
        # First pass to get the algorithm
        l_cfg, unknown = cls.parse_launcher_args(launcher_conf_parser_class)

        # Insantiate class
        launcher = cls(l_cfg)

        # Separate args for each node
        node_args = []
        for flag in unknown:
            if flag == '--':
                node_args.append([])
            elif len(node_args) > 0:
                node_args[-1].append(flag)

        assert len(node_args) == l_cfg.launcher.multi_node.num_nodes, \
            f"Node args ({len(node_args)}) does not match " \
            f"num nodes ({l_cfg.launcher.multi_node.num_nodes})"

        # Second pass to get corresponding config
        parser = alg_name_map[l_cfg.launcher.algorithm].conf_parser()()
        c_cfgs = [parser.parse_args(arg) for arg in node_args]

        # Start training
        launcher.start_training_multicli(l_cfg, c_cfgs)

    def __init__(self, cfg: Munch) -> None:
        super().__init__(cfg)

        assert len(cfg.launcher.multi_node.max_nodes_per_gpu) in {
            1, torch.cuda.device_count()}, "len(max_nodes_per_gpu) = " \
            "must be 1 of the same as number of gpu devices on this machine"

        # Allocate gpu resources
        self.gpu_resources = [
            Semaphore(cfg.launcher.multi_node.max_nodes_per_gpu[0])
            for _ in range(torch.cuda.device_count())
        ] if len(cfg.launcher.multi_node.max_nodes_per_gpu) == 1 else list(
            cfg.launcher.multi_node.max_nodes_per_gpu
        )

        self.gpu_acquizition_lock = threading.Lock()

        self.node_threads_ids_lock = Lock()
        self.node_threads_ids: List[int] = []

        # Initializes handler for the SIGINT (Ctrl-C) signal
        signal.signal(signal.SIGINT, self.sigint_handler)

    def start_training_multicli(self, launcher_cfg, node_cfgs):
        """Start all nodes and wait all of them to stop"""
        def start_node(node_class: type, cfg):
            """Create and start the algorithm node"""

            with self.node_threads_ids_lock:
                self.node_threads_ids.append(get_ident())

            assert issubclass(node_class, (ResourceManagedClient,))

            node = alg_name_map[self.algorithm](cfg)
            node.log.warning("Setting resource-related functions")
            node.acquire_resource_fn = self._acquire_gpu
            node.release_resource_fn = self._release_gpu

            # Start training
            try:
                with torch.cuda.stream(node.stream):
                    node.start_training(cfg)
            except Exception as exc:  # pylint: disable=broad-except
                node.log.exception(
                    "Exception from node captured.", exc_info=exc)
            finally:
                node.cleanup()
                # Terminal all other threads
                os.kill(os.getpid(), signal.SIGINT)

        node_threads = [Thread(
            target=start_node, args=(
                alg_name_map[launcher_cfg.launcher.algorithm], node_cfg))
            for node_cfg in node_cfgs]

        for thread in node_threads:
            thread.start()

        for thread in node_threads:
            thread.join(5)

    def sigint_handler(self, *_, **__):
        """Terminate all threads"""
        # See https://stackoverflow.com/questions/19652446
        # and https://stackoverflow.com/questions/631441
        for tid in self.node_threads_ids:
            signal.pthread_kill(tid, signal.SIGTERM)
        sys.exit(1)

    def _acquire_gpu(self, gpu_idxs: Sequence[int]):
        with self.gpu_acquizition_lock:
            for gpu_idx in gpu_idxs:
                self.gpu_resources[gpu_idx].acquire()

    def _release_gpu(self, gpu_idxs: Sequence[int]):
        for gpu_idx in gpu_idxs:
            self.gpu_resources[gpu_idx].release()


if __name__ == "__main__":
    MultiNodeLauncher.start()
