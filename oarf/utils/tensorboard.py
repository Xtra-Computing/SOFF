import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams


# from https://github.com/pytorch/pytorch/issues/32651
# hook tensorboard to remove annoying subfoler
class SummaryWriter(SummaryWriter):
    def add_hparams(self, hparam_dict, metric_dict,
                    hparam_domain_discrete=None, run_name=None):

        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError(
                'hparam_dict and metric_dict should be dictionary.')

        exp, ssi, sei = hparams(
            hparam_dict, metric_dict, hparam_domain_discrete)

        if not run_name:
            logdir = self._get_file_writer().get_logdir()
        else:
            logdir = os.path.join(
                self._get_file_writer().get_logdir(), run_name)

        with SummaryWriter(log_dir=logdir) as w_hp:
            w_hp.file_writer.add_summary(exp)
            w_hp.file_writer.add_summary(ssi)
            w_hp.file_writer.add_summary(sei)
            for k, v in metric_dict.items():
                w_hp.add_scalar(k, v)
