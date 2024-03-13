"""SplitNN utilities"""
import torch
from torch.utils.data import DataLoader
from munch import Munch
from ...datasets import create_dataset

class SplitnnDataInitializer:
    def _init_dataset(self, cfg: Munch):
        """
        Initialize test and eval dataset for the server.
        For splitnn, dataset on the server only holds link id and labels.
        Dataset on the server and clients must be aligned so that in each comm
            round the labels matches the features.
        """
        # Train dataset
        self.train_dataset = create_dataset(
            cfg, datasets=cfg.data.raw.datasets, mode='train', split_id=None)
        train_g = torch.Generator()
        train_g.manual_seed(cfg.federation.seed)
        self.train_loader = DataLoader(
            self.train_dataset, cfg.training.batch_size,
            shuffle=False, generator=train_g)
        self.train_iter = iter(self.train_loader)

        # Validate dataset
        self.eval_dataset = create_dataset(
            cfg, datasets=cfg.data.raw.datasets, mode='eval', split_id=None)
        eval_g = torch.Generator()
        eval_g.manual_seed(cfg.federation.seed)
        self.eval_loader = DataLoader(
            self.eval_dataset, cfg.training.batch_size,
            shuffle=False, generator=eval_g)
        self.eval_iter = iter(self.eval_loader)

        # Test dataset
        self.test_dataset = create_dataset(
            cfg, datasets=cfg.data.raw.datasets, mode='test', split_id=None)
        test_g = torch.Generator()
        test_g.manual_seed(cfg.federation.seed)
        self.test_loader = DataLoader(
            self.test_dataset, cfg.training.batch_size,
            shuffle=False, generator=test_g)
        self.test_iter = iter(self.test_loader)
