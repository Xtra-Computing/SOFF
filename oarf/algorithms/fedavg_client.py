import sys
import torch
import random
import oarf.compressors.compressors as compressors
from oarf.security import aes
from oarf.algorithms.base_client import BaseClient
from oarf.models.models import init_loss_criterion, init_model
from oarf.datasets.datasets import init_additional_criteria
from oarf.utils.training import (
    all_params, init_optimizer, seed_everything, init_buffer,
    evaluate_model, PerIterModelTrainer, PerEpochModelTrainer, do_svd_analysis)
from oarf.communications.protocol import (
    SyncInfos, MessageType, ClientInfo, SecureForward)


class FedAvgClient(BaseClient):
    def init_training(
            self, id, num_clients, global_momentum, error_feedback, seed,
            learning_rate, batchnorm_runstat, momentum, weight_decay,
            patience, warmup_epochs, average_policy, model, optimizer,
            datasets, secure_aggregation_method,
            # dp-related args
            dp_type, epsilon, delta, clip, epochs, **_):

        # send some metadata to aggregator
        self.dispatcher.send_msg(
            MessageType.CLIENT_INFOS, ClientInfo().set_data({
                'data_len': len(self.train_dataset),
                'histogram': self.train_dataset.histogram()}).encode())

        # the network to train on
        seed_everything(seed)
        self.net = init_model(
            model, batchnorm_runstat=batchnorm_runstat,
            dataset=self.train_dataset)
        self.net.cuda()

        # the network to preserve the parameters/gradients of the last epoch,
        # for calculating the accumulated gradient with momentum
        # (not possible with only gradient accumulation on one network)
        self.gradient = init_model(
            model, batchnorm_runstat=batchnorm_runstat,
            dataset=self.train_dataset)
        self.gradient.cuda()
        # zero-initialization is essential for fedavg
        init_buffer(self.gradient)

        # initialize buffers for splitting the gradient
        if self.secure_aggregation and secure_aggregation_method == 'SS':
            self.gradient_copy = init_model(
                model, batchnorm_runstat=batchnorm_runstat,
                dataset=self.train_dataset)
            self.gradient_copy.cuda()
            init_buffer(self.gradient_copy)

            self.random_split = init_model(
                model, batchnorm_runstat=batchnorm_runstat,
                dataset=self.train_dataset)
            self.random_split.cuda()
            init_buffer(self.random_split)

        # preserve the overall (global) gradient of the last epoch,
        # used to implement global momentum update
        if global_momentum != 0:
            self.momentum = init_model(
                model, batchnorm_runstat=batchnorm_runstat,
                dataset=self.train_dataset)
            init_buffer(self.momentum)

        # preserves the error of last compression for error feedback
        if error_feedback:
            self.error_last = init_model(
                model, batchnorm_runstat=batchnorm_runstat,
                dataset=self.train_dataset)
            init_buffer(self.error_last)

        # ensure same global model initialization in client and server
        seed_everything(seed)
        self.net_global = init_model(
            model, batchnorm_runstat=batchnorm_runstat,
            dataset=self.train_dataset)
        self.net_global.cuda()
        self.optimizer_global = torch.optim.SGD(
            self.net_global.parameters(), lr=1.0)

        self.optimizer = init_optimizer(
            optimizer, self.net.parameters(), lr=learning_rate,
            momentum=momentum, weight_decay=weight_decay)

        self.train_criterion = init_loss_criterion(model)
        self.train_criterion.cuda()
        self.additional_criteria = init_additional_criteria(datasets[0])

        self.global_lr = learning_rate
        self.global_random_seed = None

        self.num_params = len(list(self.net_global.parameters()))

        self.iters = 0
        self.epochs = 0

        self.init_trainer(average_policy, id, delta, clip)

    def init_trainer(self, average_policy, id, delta, clip):
        if average_policy == 'iter':
            self.trainer = PerIterModelTrainer(
                self.train_loader, (self.dp_type is not None),
                self.dp_noise_level, clip,
                self.tfboard_writer, "Client {}".format(id))
        elif average_policy == 'epoch':
            self.trainer = PerEpochModelTrainer(
                self.train_loader, (self.dp_type is not None),
                self.dp_noise_level, clip,
                self.tfboard_writer, "Client {}".format(id))
        else:
            raise ValueError("Unkown average policy")

    def train_one_round(self):
        return self.trainer.train_model(
            self.net, self.optimizer,
            self.train_criterion, self.additional_criteria, self.iters)

    def update_lr(self, new_lr):
        for g in self.optimizer.param_groups:
            g['lr'] = new_lr

    def handle_bye(self, msg_type):
        if msg_type == MessageType.BYE or msg_type == MessageType._BYE:
            self.log.info("Server gone. Stopping ...")
            sys.exit(0)
        # TODO: implement saving mechanism
        pass

    def start_training(
            self, id, num_clients, average_policy, server_momentum_masking,
            average_every, global_momentum, error_feedback, error_decay,
            momentum_masking, warmup_epochs, learning_rate, log_every,
            svd_analysis, lr_factor, global_learning_rate, batch_size,
            secure_aggregation_method, **_):

        with self.analyzer.training() as train_analyzer:
            train_analyzer.record(self.train_dataset.histogram())

            self.optimizer.zero_grad()
            self.optimizer.step()

            # stop when the learning rate is reduced the 4th time
            # (or server stops, whichever comes first)
            while True:
                # self.global_lr > learning_rate * (lr_factor ** 4) * 1.01:
                with self.analyzer.comm_round() as comm_round_analyzer:
                    # prevent exit failure
                    comm_round_analyzer.record(0)

                    # Receive sync information from aggregator
                    self.update_sync_info()
                    self.update_lr(self.global_lr)

                    # Shortcut if this client is not selected in this round
                    if not self.selected:
                        self.log.debug("Not updating this round")
                        continue

                    self.update_global_params(server_momentum_masking)
                    self.update_local_params()

                    # Train ###################################################
                    self.net.train()

                    self.log.info("Learning Rate: {}".format(
                        self.optimizer.param_groups[0]['lr']))

                    if self.epoch_changed(average_policy):
                        self.log.info("Epoch: {}".format(self.epochs + 1))

                    # do traininig
                    for i in range(average_every):
                        # update epoch number
                        if average_policy == 'epoch':
                            self.epochs += 1
                        elif average_policy == 'iter':
                            self.epochs = (
                                self.iters // len(self.train_loader)) + 1
                        self.iters = self.train_one_round()

                    comm_round_analyzer.record(
                        self.epochs * len(self.train_loader) +
                        self.iters % len(self.train_loader) * batch_size)

                    # Test model before aggregate per epoch.
                    # (Test of the global model is done on server)
                    if self.epoch_changed(average_policy) and  \
                            self.epochs % log_every == 0:
                        for name, loader in self.test_loaders.items():
                            evaluate_model(
                                "Test ({})".format(name), self.net, loader,
                                self.train_criterion, self.additional_criteria,
                                self.tfboard_writer,
                                "Client {} ({})".format(id, name), self.epochs)

                    self.calc_gradient()
                    self.amortize_gradient(
                        global_momentum, global_learning_rate,
                        error_feedback, error_decay, svd_analysis)
                    self.aggregate(id, num_clients, error_feedback,
                                   secure_aggregation_method)
                    self.mask_momentum(global_momentum, momentum_masking)

    def epoch_changed(self, average_policy):
        return average_policy == 'epoch' or (
            average_policy == 'iter' and
            (self.iters // len(self.train_loader)) + 1 > self.epochs)

    def update_sync_info(self):
        with self.analyzer.recv_data() as recv_data_analyzer:
            msg_type, data = self.dispatcher.recv_msg()
            recv_data_analyzer.record(len(data))
        self.handle_bye(msg_type)
        assert (msg_type == MessageType.SYNC_INFOS)

        sync_info = SyncInfos().decode(data)
        self.global_lr = sync_info.lr
        self.global_random_seed = sync_info.seed
        self.selected = sync_info.selected
        seed_everything(self.global_random_seed)    # sync global seed

        # Update compressor parameters
        if isinstance(self.s_compressor, compressors._RandomCompressor):
            self.s_compressor.seed = random.random()
        if isinstance(self.c_compressor, compressors._RandomCompressor):
            self.c_compressor.seed = random.random()

    def update_global_params(self, server_momentum_masking):
        with self.analyzer.recv_data() as recv_data_analyzer:
            msg_type, data = self.dispatcher.recv_msg()
            recv_data_analyzer.record(len(data))
        self.handle_bye(msg_type)
        if msg_type == MessageType.GRADIENT:
            # update global net using global grad
            with self.analyzer.decompress():
                gradients = self.s_compressor.decompress(data)

            self.net_global.train()
            # update parameters and buffers
            for param, grad in zip(self.net_global.parameters(),
                                   gradients[:self.num_params]):
                param.grad = grad.cuda()
            for buf, grad in zip(self.net_global.buffers(),
                                 gradients[self.num_params:]):
                buf.copy_(buf - grad.cuda())
            self.optimizer_global.step()
            self.net_global.eval()

            # perform server momentum masking. The mask can only be
            # calculated after receiving server momentum
            if server_momentum_masking:
                self.s_compressor.zero_with_reference(self.momentum, gradients)
        elif msg_type == MessageType.MODEL:
            with self.analyzer.decompress():
                params = self.s_compressor.decompress(data)
            with torch.no_grad():
                for g_param, param in zip(
                        all_params(self.net_global), params):
                    g_param.set_(param.cuda())
        else:
            raise Exception("Client can only handle gradient or model.")

    def update_local_params(self):
        # update local gradient using global net, so new parameters could be
        # substracted from it later
        for grad, param_g in zip(all_params(self.gradient),
                                 all_params(self.net_global)):
            grad.copy_(param_g)

        for param, param_g in zip(all_params(self.net),
                                  all_params(self.net_global)):
            param.data.copy_(param_g)

    def calc_gradient(self):
        """calculate gradient and store into self.gradient"""
        self.log.info("Calculating gradients...")

        # calculate gradient and store in `self.gradient`
        for grad, param in zip(all_params(self.gradient),
                               all_params(self.net)):
            grad.copy_(grad - param)

    def amortize_gradient(self, global_momentum, global_learning_rate,
                          error_feedback, error_decay, svd_analysis):
        # add global momentum.
        # no momentum as warmup epoch, otherwise will cause damage
        if global_momentum != 0:
            # no gobal momentum for batchnorm
            for grad, mom in zip(self.gradient.parameters(),
                                 self.momentum.parameters()):
                grad.add_(mom)

        # TODO: global momentum should be implemented on server
        # since decompress a momentumized gradient (not sparse) may cause
        # more err (i.e. pass parameter to optimizer)
        for grad in self.gradient.parameters():
            grad.mul_(global_learning_rate)

        # apply error feedback, if necessary
        if error_feedback:
            for grad, err_l in zip(all_params(self.gradient),
                                   all_params(self.error_last)):
                grad.add_((err_l * (1.0 - error_decay)).type(grad.dtype))

        # analyze gradient
        if svd_analysis:
            do_svd_analysis(self.gradient, self.tfboard_writer,
                            "Client {}".format(id), self.epochs)

    def aggregate(
            self, id, num_clients, error_feedback, secure_aggregation_method):
        # update error for error feedback in the next rond
        data = None
        if error_feedback:
            data = self.c_compressor.compress(all_params(self.gradient))
            with self.analyzer.decompress():
                decompressed = self.c_compressor.decompress(data)
            for err_l, grad, grad_d in zip(all_params(self.error_last),
                                           all_params(self.gradient),
                                           decompressed):
                err_l.copy_(grad - grad_d.cuda())

        if not self.secure_aggregation:
            """sent gradient info to aggregator"""
            with self.analyzer.compress():
                data = (data if data is not None else
                        self.c_compressor.compress(all_params(self.gradient)))
            # send gradient to server
            self.log.info("Sending gradient ({} bytes)".format(len(data)))
            with self.analyzer.send_data() as send_data:
                self.dispatcher.send_msg(MessageType.GRADIENT, data)
                send_data.record(bytes_sent=len(data))
        else:
            if secure_aggregation_method == "SS":
                # TODO: Ensure upper bound is larger than
                # 2 * max(|element in grad|). so currently we simply use a
                # large enough value
                UPPER_BOUND = 9973.0
                self.gradient_copy.load_state_dict(self.gradient.state_dict())

                # offset all values to positive number
                for grad in self.gradient_copy.parameters():
                    grad.add_(UPPER_BOUND/2)
                    assert (grad > 0).all() and (grad < UPPER_BOUND).all()

                # split grdient and sent to other clients
                for i in range(self.secret_splitting_num - 1):
                    for grad, rand in zip(self.gradient_copy.parameters(),
                                          self.random_split.parameters()):
                        rand.set_(torch.rand_like(grad).cuda() * UPPER_BOUND)
                        grad.add_(UPPER_BOUND)
                        grad.subtract_(rand)
                        grad.fmod_(UPPER_BOUND)
                    # pack and encrypt with random aes key
                    data = compressors.NoCompress().compress(
                        list(self.random_split.parameters()))
                    aes_key = aes.Key(gen_key=True)
                    data = aes_key.encrypt(data)
                    dst = (id + i + 1) % num_clients
                    key = self.rsa_keyring[dst].encrypt(aes_key.seralize_key())

                    # send data to other clients
                    data = SecureForward().set_data({
                        'src': id, 'dst': dst,
                        'key': key, 'data': data}).encode()
                    self.log.info("Sending split to client {} ({} bytes)"
                                  .format(dst, len(data)))
                    with self.analyzer.send_data() as send_data:
                        send_data.record(len(data))
                        self.dispatcher.send_msg(MessageType.GRAD_SPLIT, data)

                # receive splits from other clients
                for i in range(self.secret_splitting_num - 1):
                    with self.analyzer.recv_data() as recv_data:
                        msg_type, data = self.dispatcher.recv_msg()
                        recv_data.record(len(data))
                    assert msg_type == MessageType.GRAD_SPLIT, msg_type
                    data = SecureForward().decode(data)
                    self.log.info("Received split from client {}".format(
                        data.src))

                    aes_key = aes.Key()
                    aes_key.deseralize_key(self.rsa_key.decrypt(data.key))
                    split = compressors.NoCompress().decompress(
                        bytearray(aes_key.decrypt(data.data)))
                    for g_c, sp in zip(self.gradient_copy.parameters(), split):
                        g_c.add_(sp.cuda())
                        g_c.fmod_(UPPER_BOUND)

                # offset the values back to normal
                for grad in self.gradient_copy.parameters():
                    grad.subtract_(UPPER_BOUND / 2)

                # send pre-aggregated data to sever
                with self.analyzer.compress():
                    data = self.c_compressor.compress(
                        all_params(self.gradient_copy))
                self.log.info("Sending gradient ({} bytes)".format(len(data)))
                with self.analyzer.send_data() as send_data:
                    send_data.record(len(data))
                    self.dispatcher.send_msg(MessageType.GRADIENT, data)
            else:
                raise NotImplementedError("Not implemented yet")

    def mask_momentum(self, global_momentum, momentum_masking):
        """perform momentum masking"""
        if global_momentum != 0:
            for mom, grad in zip(self.momentum.parameters(),
                                 self.gradient.parameters()):
                mom.mul_(global_momentum).add_(grad)
            if momentum_masking:
                self.c_compressor.zero_with_mask(mom)
