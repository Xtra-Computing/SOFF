import argparse
import numpy as np
import os
import sys
import yaml

from dcrnn_train_pytorch import DCRNNSupervisor


def run_dcrnn(args):

    supervisor = DCRNNSupervisor()
    supervisor.load_model(64)

    # VLD
    val_loss1, dict1, val_loss2, dict2 = supervisor.evaluate(dataset='val')
    mae1 = supervisor.mae_at_12(dict1['prediction'], dict1['truth'])
    mae2 = supervisor.mae_at_12(dict2['prediction'], dict2['truth'])
    val_loss = (val_loss1 + val_loss2) / 2
    message = 'VAL val_loss_la: {:.2f}, val_loss_bay: {:.2f}, val_mae12_la: {:.2f}, val_mae12_bay: {:.2f}' \
            .format(val_loss1, val_loss2, mae1, mae2)
    print(message)

    # TST
    tst_loss1, dict1, tst_loss2, dict2 = supervisor.evaluate(dataset='test')
    mae1 = supervisor.mae_at_12(dict1['prediction'], dict1['truth'])
    mae2 = supervisor.mae_at_12(dict2['prediction'], dict2['truth'])
    message = 'TEST tst_loss_la: {:.2f}, tst_loss_bay: {:.2f}, tst_mae12_la: {:.2f}, tst_mae12_bay: {:.2f}'.format(tst_loss1, tst_loss2, mae1, mae2)
    print(message)


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cpu_only', default=False, help='Whether to run tensorflow on cpu.')
    parser.add_argument('--config_filename', default='data/model/pretrained/METR-LA/config.yaml')
    parser.add_argument('--output_filename', default='data/dcrnn_predictions.npz')
    args = parser.parse_args()
    run_dcrnn(args)