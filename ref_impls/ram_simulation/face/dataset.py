import h5py as h5
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import RandomSampler, DataLoader, Dataset

def get_datasets(args):
    # load data
    class FR(Dataset):  # Training Dataset
        def __init__(self, party, group):
            path = '{0}/data/{1}_train_val.h5'.format(args.task_dir, party)
            self.archive = h5.File(path, 'r')
            self.x = self.archive['x_'+group][:]
            self.y = self.archive['y_'+group][:]

            self.trn = True if group == 'train' else False
            self.party = party

            if self.trn:
                self.transforms = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip()
                ])

        def __getitem__(self, index):
            datum = self.x[index].copy()
            datum = torch.from_numpy(datum)
            if self.trn:
                datum = self.transforms(datum)
                datum = torch.from_numpy(np.array(datum))
                datum = datum.permute((2, 0, 1))
            datum = (datum - 127.5) / 128.0
            label = self.y[index].copy().astype('int64')
            if args.setting == 'fedavg' or args.setting == 'combined':
                if self.party == 'African':
                    label += 7000
            return datum, label

        def __len__(self):
            if self.trn and args.dp:
                return len(self.y) - len(self.y) % args.batch_size
            else:
                return len(self.y)

        def close(self):
            self.archive.close()
    
    
    class RFW(Dataset):  # Test Dataset
        def __init__(self, party):
            archive = h5.File('data/{0}_test.h5'.format(party), 'r')

            self.img1 = archive['img1'][:]
            self.img1_flip = archive['img1_flip'][:]
            self.img2 = archive['img2'][:]
            self.img2_flip = archive['img2_flip'][:]
            self.labels = archive['labels'][:]

        def __getitem__(self, index):
            i1 = self.process(self.img1[index])
            i1f = self.process(self.img1_flip[index])
            i2 = self.process(self.img2[index])
            i2f = self.process(self.img2_flip[index])
            label = self.labels[index]

            return i1, i1f, i2, i2f, label

        def process(self, img):
            datum = img.copy()
            datum = torch.from_numpy(datum)
            datum = (datum - 127.5) / 128.0
            return datum

        def __len__(self):
            return len(self.img1)

        def close(self):
            self.archive.close()


    trn_cau = FR('Caucasian', 'train')
    trn_afr = FR('African' , 'train')
    val_cau = FR('Caucasian', 'val')
    val_afr = FR('African', 'val')
    tst_cau = RFW('Caucasian')
    tst_afr = RFW('African')
   
    trn_party_datasets = [trn_cau, trn_afr]
    val_party_datasets = [val_cau, val_afr]
    tst_party_datasets = [tst_cau, tst_afr]
    
    return trn_party_datasets, val_party_datasets, tst_party_datasets


