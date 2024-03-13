"""Full-feature FedAvg algorithm server"""

import math
from .. import fedavg
from ..base.ff_base_server import FFBaseServer, FFBaseServerConfParser
from ...communications.protocol import MessageType, Protocol
from ...compressors.none import NoCompress
from ...utils.training import all_params
from ...models.base import PerEpochTrainer, PerIterTrainer
from ...models import ModelTrainerConfParser, model_trainer_name_map


class ConfParser(
        ModelTrainerConfParser,
        FFBaseServerConfParser,
        fedavg.ConfParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        fffedavg_args = self.add_argument_group(
            "Full-feature FedAvg Configs (S,S->C)")
        fffedavg_args.add_argument(
            '-ffavg.bt', '--fffedavg.broadcast-type',
            default='model', choices=['gradient', 'model'],
            help="Server broadcast type. 'gradient' can save communication, "
            "since it may benefit from server gradient compressor, "
            "but can only be used when client-fraction = 1 (e.g. fedsgd), "
            "since we need to ensure that global models are in sync.")


class Server(fedavg.Server, FFBaseServer):
    """
    Full-feature FedAvg algorithm server, supports broadcasting gradient and
    per-iter aggregation.
    """
    @classmethod
    def conf_parser(cls):
        return ConfParser

    def __init__(self, cfg):
        super().__init__(cfg)

        client_fraction = cfg.fedavg.client_fraction
        broadcast_type = cfg.fffedavg.broadcast_type

        assert client_fraction == 1 or broadcast_type != 'gradient', \
            "broadcast_type can only be 'model' when client fraction is not 1"

        self.trainer_class = model_trainer_name_map[cfg.training.model_trainer.name]
        self.log.info("Average policy: %s", cfg.training.model_trainer.name)

    def schedule_broadcast_model_data(self) -> None:
        # Deal with both grads/models, sending only to selected clients

        if self.cfg.fffedavg.broadcast_type == "gradient":
            data = NoCompress().compress(all_params(self.grads))
            msg_type = MessageType.GRADIENT
        elif self.cfg.fffedavg.broadcast_type == "model":
            data = NoCompress().compress(all_params(self.net))
            msg_type = MessageType.MODEL
        else:
            raise Exception("Unknown broadcast_type")

        self.log.info(
            "  Broadcasting %s to %s (%s bytes × %s clients)",
            self.cfg.fffedavg.broadcast_type, self.selected_client_ids,
            len(data), len(self.selected_client_ids))

        # broadcasting gradient
        for client_id in self.selected_client_ids:
            socket = self.scheduler.clients_id_socket_map[client_id]
            self.scheduler.dispatcher.schedule_task(
                Protocol.send_data, socket, msg_type, data,
                self.datalogger, self.epoch)

        # fire event once everything is sent
        self.scheduler.dispatcher.insert_task_barrier()

    def calc_total_comm_rounds(self) -> int:
        mt_name = self.cfg.training.model_trainer.name
        if issubclass(model_trainer_name_map[mt_name], PerEpochTrainer):
            return super().calc_total_comm_rounds()

        if issubclass(model_trainer_name_map[mt_name], PerIterTrainer):
            return self.cfg.training.epochs * (
                self.scheduler.sum_data_lengths // (
                    self.cfg.fedavg.average_every *
                    self.cfg.training.batch_size *
                    self.cfg.client_server.num_clients) + 1)

        raise Exception("Unknown average policy")

    def update_equivalent_epoch_number(self) -> None:
        """Calculate and update the equivalent epoch number """
        mt_name = self.cfg.training.model_trainer.name
        if issubclass(model_trainer_name_map[mt_name], PerEpochTrainer):
            super().update_equivalent_epoch_number()
            return

        assert issubclass(model_trainer_name_map[mt_name], PerIterTrainer)

        self.old_epoch = self.epoch
        sum_samples = 0
        for cli_id in self.scheduler.clients_id_socket_map.keys():
            dataset_len = self.scheduler.clients_dataset_length[cli_id]
            iters = (
                self.clients_comm_rounds[cli_id] *
                self.cfg.fedavg.average_every)
            iters_per_epoch = math.ceil(
                dataset_len / self.cfg.training.batch_size)
            sum_samples += (
                (iters // iters_per_epoch) * dataset_len +
                (iters % iters_per_epoch) * self.cfg.training.batch_size)
        self.epoch = sum_samples // self.scheduler.sum_data_lengths
