import h5py as h5
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import RandomSampler, DataLoader, Dataset

def get_datasets(args):
    # load data
    class ACR(Dataset):  # Alphanumeric Character Recognition Dataset
        def __init__(self, archive, group):
            self.trn = True if group == 'train' else False
            self.archive = h5.File(archive, 'r')
            self.x = self.archive['x_'+group]
            self.y = self.archive['y_'+group]

        def __getitem__(self, index):
            datum = self.x[index]
            datum = torch.from_numpy(datum).float().div(255)
            label = self.y[index].astype('int64')
            return datum, label

        def __len__(self):
            if self.trn and args.dp:
                return len(self.y) - len(self.y) % args.batch_size
            else:
                return len(self.y)

        def close(self):
            self.archive.close()

    path_fnt = args.task_dir + '/data/fnt.h5'
    path_hnd = args.task_dir + '/data/hnd.h5'

    trn_fnt = ACR(path_fnt, 'train')
    trn_hnd = ACR(path_hnd , 'train')
    val_fnt = ACR(path_fnt, 'val')
    val_hnd = ACR(path_hnd, 'val')
    tst_fnt = ACR(path_fnt, 'test')
    tst_hnd = ACR(path_hnd, 'test')
   
    trn_party_datasets = [trn_fnt, trn_hnd]
    val_party_datasets = [val_fnt, val_hnd]
    tst_party_datasets = [tst_fnt, tst_hnd]
    
    return trn_party_datasets, val_party_datasets, tst_party_datasets