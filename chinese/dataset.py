import h5py as h5
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import RandomSampler, DataLoader, Dataset

def get_datasets(args):
    
    class HCC(Dataset):  # Handwritten Chinese character dataset
        # group: trn/vld
        def __init__(self, archive, group, transform=None):
            self.trn = True if group == 'trn' else False
            self.archive = h5.File(archive, 'r')
            if args.debug:
                self.x = self.archive[group + '/x'][:1280]
                self.y = self.archive[group + '/y'][:1280]
            else:
                self.x = self.archive[group + '/x']
                self.y = self.archive[group + '/y']
            self.transform = transform
        def __getitem__(self, index):
            datum = self.x[index]  # numpy array (C,H,W)
            datum = torch.from_numpy(datum)  # tensor (C,H,W)
            if self.transform is not None:
                datum = self.transform(datum)
            label = self.y[index][0].astype('int64')
            return datum, label
        def __len__(self):
            if self.trn and args.dp:
                return len(self.y) - len(self.y) % args.batch_size
            else:
                return len(self.y)
        def close(self):
            self.archive.close()
        
    def to_tensor(img):
        return img.float().div(255)
    tfm = transforms.Lambda(to_tensor)
    
    casia = args.task_dir + '/data/HWDB1.1.hdf5'
    hit = args.task_dir + '/data/HIT_OR3C.hdf5'

    trnset_casia = HCC(casia, 'trn', transform=transforms.Compose([tfm]))
    trnset_hit = HCC(hit, 'trn', transform=transforms.Compose([tfm]))
    valset_casia = HCC(casia, 'vld', transform=transforms.Compose([tfm]))
    valset_hit = HCC(hit, 'vld', transform=transforms.Compose([tfm]))
    tstset_casia = HCC(casia, 'tst', transform=transforms.Compose([tfm]))
    tstset_hit = HCC(hit, 'tst', transform=transforms.Compose([tfm]))
    
    trn_party_datasets = [trnset_casia, trnset_hit]
    val_party_datasets = [valset_casia, valset_hit]
    tst_party_datasets = [tstset_casia, tstset_hit]
    
    return trn_party_datasets, val_party_datasets, tst_party_datasets
