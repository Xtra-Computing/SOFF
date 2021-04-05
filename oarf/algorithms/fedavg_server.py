import math
import torch
import random
import threading
import oarf.compressors.compressors as compressors
from eventfd import EventFD
from functools import partial
from oarf.algorithms.base_server import BaseSever
from oarf.utils.model_buffer import ModelBuffer
from oarf.utils.scheduler import GradualWarmupScheduler
from oarf.models.models import init_loss_criterion, init_model
from oarf.datasets.datasets import init_additional_criteria
from oarf.communications.protocol import (
    SyncInfos, MessageType, Protocol, ClientInfo)
from oarf.utils.training import (
    all_params, init_optimizer, seed_everything, init_buffer, evaluate_model)


class FedAvgServer(BaseSever):
    def __init__(self, num_clients, client_fraction, broadcast_type, **kwargs):
        super().__init__(**kwargs, num_clients=num_clients,
                         client_fraction=client_fraction,
                         broadcast_type=broadcast_type)

        if broadcast_type == 'gradient' and client_fraction != 1:
            raise Exception("broadcast_type can only be 'model' \
                            when client fraction is not 1")
        if broadcast_type == 'model' and not (
                isinstance(self.s_compressor, compressors.NoCompress) or
                isinstance(self.s_compressor, compressors.SparseCompress)):
            raise Exception(
                "broadcasting model can only use non-lossy compressor")

        self.num_clients_each_round = \
            max(round(num_clients * client_fraction), 1)

        self.event_aggregation_done = EventFD()
        """ signal the global model is aggregated and ready to be send"""
        self.event_broadcast_done = threading.Event()
        """signals the gradients broadcast is done, and training can proceed"""

        self.clients_dataset_length = {}
        self.clients_dataset_histos = {}
        self.clients_dataset_lock = threading.Lock()
        """record each clients' dataset length"""

        self.event_training_started = threading.Event()
        """This event is fired once all clients' info are received"""

        self.dispatcher.register_msg_event(
            MessageType.CLIENT_INFOS,
            partial(self.process_client_info, num_clients=num_clients))
        self.dispatcher.register_msg_event(
            MessageType.GRADIENT, self.schedule_process_gradient)
        self.dispatcher.register_fd_event(
            self.event_aggregation_done,
            partial(self.schedule_broadcast_model,
                    broadcast_type=broadcast_type))

    def process_client_info(self, socket, data, /, num_clients):
        client_info = ClientInfo().decode(data)
        client_id = self.clients_socket_id_map[socket]

        with self.clients_dataset_lock:
            self.clients_dataset_length[client_id] = client_info.data_len
            self.clients_dataset_histos[client_id] = client_info.histogram
            self.log.info("  Client {} dataset length: {}".format(
                client_id, client_info.data_len))

            if len(self.clients_dataset_length.keys()) == num_clients:
                self.sum_data_lengths = sum(
                    self.clients_dataset_length.values())
                self.event_training_started.set()

    def schedule_process_gradient(self, socket, data):
        def process_gradient(data):
            # skip processing if client already disconnected
            with self.clients_info_lock:
                if socket in self.clients_socket_id_map:
                    # put data into slot
                    idx, net_client = \
                        self.client_nets_buffer.get_net_for_receive()
                    net_client.set_grad(self.c_compressor.decompress(data))
                    net_client.id = self.clients_socket_id_map[socket]
                    self.client_nets_buffer.release_as_ready(idx)

                    self.log.info(
                        "  Gradient of client {} ✔".format(net_client.id))
                    self.sem_client_event.release()
        self.dispatcher.schedule_task(process_gradient, data)

    def schedule_broadcast_model(self, _, /, broadcast_type):
        self.log.info("  Broadcasting model and related infos...")
        self.schedule_broadcast_sync_info()
        self.schedule_broadcast_gradient(broadcast_type)
        self.dispatcher.schedule_task(self.event_broadcast_done.set)

    def schedule_broadcast_sync_info(self):
        sync_info = SyncInfos().set_data({
            'lr': self.global_optimizer.param_groups[0]['lr'],
            'seed': self.global_random_seed})

        # Bcast sync info
        self.log.info("  Broadcasting sync infos")
        for id, socket in self.clients_id_socket_map.items():
            sync_info.selected = (id in self.selected_client_ids)
            self.dispatcher.schedule_task(
                Protocol.send_data, socket,
                MessageType.SYNC_INFOS, sync_info.encode(),
                self.analyzer.send_data_parallel())

        # must wait for syncinfo to finish, before sending gradient
        self.dispatcher.insert_task_barrier()

    def schedule_broadcast_gradient(self, broadcast_type):
        # deal with both grads/models, sending only to selected clients
        if broadcast_type == "gradient":
            with self.analyzer.compress():
                data = self.s_compressor.compress(all_params(self.grads))
            msg_type = MessageType.GRADIENT
        elif broadcast_type == "model":
            with self.analyzer.compress():
                data = self.s_compressor.compress(all_params(self.net))
            msg_type = MessageType.MODEL
        else:
            raise Exception("Unknown broadcast_type")

        self.log.info(
            "  Broadcasting {} to {} ({} bytes × {} clients)".format(
                broadcast_type, self.selected_client_ids,
                len(data), len(self.selected_client_ids)))

        # broadcasting gradient
        for client_id in self.selected_client_ids:
            socket = self.clients_id_socket_map[client_id]
            self.dispatcher.schedule_task(
                Protocol.send_data, socket, msg_type, data,
                self.analyzer.send_data_parallel())

        # fire event once everything is sent
        self.dispatcher.insert_task_barrier()

    def schedule_broadcast_bye(self):
        self.log.info("  Broadcasting shutdown signal")
        self.dispatcher.schedule_broadcast(
            MessageType.BYE, data=b'Bye!', analyzer=self.analyzer)
        self.dispatcher.insert_task_barrier()
        self.dispatcher.schedule_task(self.event_broadcast_done.set)

    def init_training(
            self, model, datasets, seed, batchnorm_runstat, num_cache,
            learning_rate, patience, warmup_epochs, lr_factor, optimizer, **_):
        # Initialize server network ###########################################
        # server network, ensure same global model initialization in C/S
        seed_everything(seed)
        self.net = init_model(
            model, batchnorm_runstat=batchnorm_runstat,
            dataset=self.eval_dataset)
        self.net.cuda()
        self.num_params = len(list(self.net.parameters()))
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=1.0)

        # Gradient of `net`, stored in the parameters of a model.
        # Must be zeroed out so that the first epoch is correct.
        self.grads = init_model(
            model, batchnorm_runstat=batchnorm_runstat,
            dataset=self.eval_dataset)
        init_buffer(self.grads)

        self.train_criterion = init_loss_criterion(model)
        self.train_criterion.cuda()
        self.additional_criteria = init_additional_criteria(datasets[0])

        # Initialize trainig ##################################################
        # a list of client networks to receive
        self.client_nets_buffer = ModelBuffer(num_cache, self.net)

        # Then, start then trainig process
        self.global_model_loss = 10.0
        self.global_random_seed = random.randint(0, 2 ** 31)

        # dummy optimizer and schedular for global lr scheduling
        self.global_optimizer = init_optimizer(
            optimizer, [torch.Tensor()], lr=learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.global_optimizer, mode='min', factor=lr_factor,
            patience=patience, verbose=True, threshold=0.001)
        self.lr_scheduler_withwarmup = self.lr_scheduler \
            if warmup_epochs == 0 \
            else GradualWarmupScheduler(
                self.global_optimizer, multiplier=1, total_epoch=warmup_epochs,
                after_scheduler=self.lr_scheduler)

    def start_training(
            self, epochs, batch_size, learning_rate, lr_factor,
            num_clients, average_policy, average_every,
            log_every, broadcast_type, **_):
        with self.analyzer.training() as train_analyzer:
            # First, wait for all clients to connect
            self.log.info(
                "Waiting for {} clients to connect..."
                .format(num_clients))
            self.event_all_clients_connected.wait()
            self.log.info(
                "Waiting for {} clients to send metadatas..."
                .format(num_clients))
            self.event_training_started.wait()
            train_analyzer.record(list(self.clients_dataset_histos.values()))

            self.clients_comm_rounds = {
                id: 0 for id in self.clients_id_socket_map.keys()}

            # equivalent epoch number, calculated by
            # (number_samples_processed / total_number_samples)
            self.epoch = 0

            # wait for init broadcast to complete before starting other tasks
            self.dispatcher.insert_task_barrier()

            if average_policy == 'epoch':
                total_rounds = epochs
            elif average_policy == 'iter':
                total_rounds = epochs * (
                    self.sum_data_lengths // (average_every * batch_size) + 1)
            else:
                raise Exception("Unknown average policy")

            # start the main training loop
            for e in range(total_rounds):
                # notify client to exit if learning rate is small enough
                if self.global_optimizer.param_groups[0]['lr'] < \
                        learning_rate * (lr_factor ** 4) * 1.01:
                    self.schedule_broadcast_bye()
                    self.event_broadcast_done.wait()
                    break

                with self.analyzer.comm_round() as comm_round_analyzer:

                    self.init_comm_round(e)
                    with self.analyzer.broadcast() as broadcast_analyzer:
                        broadcast_analyzer.record(self.num_clients_each_round)
                        self.broadcast_model(broadcast_analyzer)

                        # wait for broadcast to finish. Before that,
                        # avoid changing gradient or selected_client_ids
                        self.event_broadcast_done.wait()
                        self.event_broadcast_done.clear()

                    with self.analyzer.aggregate() as aggregate_analyzer:
                        aggregate_analyzer.record(self.num_clients_each_round)
                        self.aggregate(num_clients)
                        self.update_global_model(broadcast_type)
                        self.update_equivalent_epoch_number(
                            average_policy, average_every,
                            batch_size, comm_round_analyzer)

                    self.eval_model(comm_round_analyzer)
                    self.adjust_lr()

                    self.test_model(e, log_every)
                    self.update_random_seed()

                    # # return if all clients are gone
                    # self.handle_shutdown_events()
            self.log.info("Trainig finished. Stopping...")

    def init_comm_round(self, e):
        self.log.info("====================================")
        self.log.info("Communication Round {}".format(e + 1))
        # update compressor parameters per communication round
        seed_everything(self.global_random_seed)
        if isinstance(self.s_compressor, compressors._RandomCompressor):
            self.s_compressor.seed = random.random()
        if isinstance(self.c_compressor, compressors._RandomCompressor):
            self.c_compressor.seed = random.random()

    def broadcast_model(self, broadcast_event):
        # select a faction of clients randomly, we use unifrom sample
        # without replacement, then average by datasize as weight
        self.selected_client_ids = set(random.sample(
            self.clients_id_socket_map.keys(),
            min(self.num_clients_each_round,
                len(self.clients_id_socket_map.keys()))))

        # send network
        self.log.info("  Broadcasting network parameters...")
        self.broadcast_event = broadcast_event
        self.event_aggregation_done.set()

    def aggregate(self, num_clients):
        # zero out all params in grads for client aggregation
        for param in all_params(self.grads):
            param.zero_()

        self.log.info("  Waiting for clients input...")
        selected_data_length = sum([self.clients_dataset_length[id]
                                    for id in self.selected_client_ids])
        clients_aggregated = [False for _ in range(num_clients)]
        for i in range(len(self.selected_client_ids)):
            # handle client events
            self.sem_client_event.acquire()
            # decide the first ready net
            # idx is the index of `client_nets_buffer` array
            idx, net_client = \
                self.client_nets_buffer.get_net_for_aggregate()

            # report if a client already aggregated by appeared again
            if clients_aggregated[net_client.id]:
                raise Exception(
                    "Client {} is not obeying the {} protocol".format(
                        net_client.id, self.__class__.__name__))

            # aggregation
            self.clients_comm_rounds[net_client.id] += 1
            client_weight = self.clients_dataset_length[net_client.id] / \
                selected_data_length
            self.log.info("  Aggregating client {} (weight {})".format(
                net_client.id, client_weight))

            # update parameter and buffers gradients
            for grad, client_grad in zip(
                    all_params(self.grads), net_client.grad):
                # update parameter grad
                grad.set_(
                    ((grad if (grad is not None)
                      else torch.zeros_like(client_grad)) +
                     client_grad.cuda() * client_weight)
                    .type(grad.dtype).clone())

            # release the net to make a vacancy for following receive
            self.client_nets_buffer.release_as_idle(idx)

    def update_global_model(self, broadcast_type):
        if broadcast_type == 'gradient' \
                and not \
                isinstance(self.s_compressor, compressors.SparseCompress) \
                and not \
                isinstance(self.s_compressor, compressors.NoCompress):
            # If server uses lossy compression, make sure the server side
            # and clientside uses the same weight to update the global one.
            self.log.debug(
                "Using compress-decompress process to match client grad.")
            with self.analyzer.compress():
                compressed_grad = \
                    self.s_compressor.compress(all_params(self.grads))
            with self.analyzer.decompress():
                decompressed_grad = \
                    self.s_compressor.decompress(compressed_grad)
        else:
            decompressed_grad = all_params(self.grads)

        # update global model
        self.net.train()
        # update parameter and buffers. use copy here,
        # otherwise `grads` will be zeroed out unexpectedly
        for param, grad in zip(self.net.parameters(),
                               decompressed_grad[:self.num_params]):
            param.grad = grad.clone()
        for param, grad in zip(self.net.buffers(),
                               decompressed_grad[self.num_params:]):
            param.copy_((param if (param is not None)
                         else torch.zeros_like(grad)) - grad)
        self.optimizer.step()
        self.net.eval()

    def update_equivalent_epoch_number(self, average_policy, average_every,
                                       batch_size, comm_round_analyzer):
        # calculate equivalent epoch number
        self.old_epoch = self.epoch
        if average_policy == 'epoch':
            sum_samples = sum([self.clients_comm_rounds[id] * average_every *
                               self.clients_dataset_length[id]
                               for id in self.clients_id_socket_map.keys()])
        elif average_policy == 'iter':
            sum_samples = 0
            for id in self.clients_id_socket_map.keys():
                dataset_len = self.clients_dataset_length[id]
                iters = self.clients_comm_rounds[id] * average_every
                iters_per_epoch = math.ceil(dataset_len / batch_size)
                sum_samples += ((iters // iters_per_epoch) * dataset_len +
                                (iters % iters_per_epoch) * batch_size)
        else:
            raise Exception("Unknown average policy")

        comm_round_analyzer.num_samples = sum_samples
        self.epoch = sum_samples // self.sum_data_lengths

    def eval_model(self, comm_round_analyzer):
        # eval
        self.global_model_loss, additional_metrics = evaluate_model(
            "Eval", self.net, self.eval_loader,
            self.train_criterion, self.additional_criteria,
            self.tfboard_writer, "Server", self.epoch)
        comm_round_analyzer.eval_loss = self.global_model_loss
        comm_round_analyzer.eval_additional_metrics = additional_metrics

    def adjust_lr(self):
        # adjust lr
        if self.epoch > self.old_epoch:
            self.lr_scheduler_withwarmup.step(
                metrics=self.global_model_loss)
            self.log.info("Equivalent epoch: {}, lr: {}".format(
                self.epoch, self.global_optimizer.param_groups[0]['lr']))

    def test_model(self, e, log_every):
        # test
        if e % log_every == 0:
            for name, loader in self.test_loaders.items():
                self.test_loss, self.additional_metrics = evaluate_model(
                    "Test ({})".format(name), self.net, loader,
                    self.train_criterion, self.additional_criteria,
                    self.tfboard_writer, "Server ({})".format(name),
                    self.epoch)

    def update_random_seed(self):
        # update global random seed,
        seed_everything(self.global_random_seed)
        self.global_random_seed = random.randint(0, 2 ** 31)
        self.log.info("  Updated random seed: {}".format(
            self.global_random_seed))

    def cleanup_hook(self, args):
        # log current run results
        if self.tfboard_writer:
            self.tfboard_writer.add_hparams(
                {k: v for k, v in args.items() if isinstance(
                    v, (int, float, str, bool, torch.Tensor))},
                {"loss": self.test_loss,
                    **{self.additional_criteria[i].__name__: c
                        for i, c in enumerate(self.additional_metrics)}},
                run_name="")
