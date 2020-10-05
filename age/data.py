import numpy as np
import torch
import PIL
from torchvision import transforms

input_size=224
image_transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class Dataset():
    def __init__(self, features_path, labels_path):
        self.features = np.load(features_path)
        self.labels = np.load(labels_path)
        assert(len(self.features) == len(self.labels))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.features[idx]
        image = PIL.Image.open(img_path)
        # convert grayscale to rgb
        if image.mode == 'L':
            rgbImage = PIL.Image.new("RGB", image.size)
            rgbImage.paste(image)
            image = rgbImage

        return image_transform(image), self.labels[idx]

class Dataset_ImdbWiki(Dataset):
    def __init__(self, data_dir):
        super().__init__(data_dir + '/imdbWikiFeatures.npy', data_dir + '/imdbWikiLabels.npy')

class Dataset_AllAgeFaces(Dataset):
    def __init__(self, data_dir):
        super().__init__(data_dir + '/allAgeFacesFeatures.npy', data_dir + '/allAgeFacesLabels.npy')

class Dataset_APPA(Dataset):
    def __init__(self, data_dir):
        super().__init__(data_dir + '/appaFeatuers.npy', data_dir + '/appaLabels.npy')

class FusionDatset(Dataset):
    def __init__(self, data_paths):
        self.features = np.array([])
        self.labels = np.array([])
        for features_path, labels_path in data_paths:
            self.features = np.concatenate([self.features, np.load(features_path)])
            self.labels = np.concatenate([self.labels, np.load(labels_path)])
        assert(len(self.features) == len(self.labels))

class Dataset_AllAgeFaces_APPA(FusionDatset):
    def __init__(self, data_dir):
        super().__init__([
            (data_dir + '/allAgeFacesFeatures.npy', data_dir + '/allAgeFacesLabels.npy'),
            (data_dir + '/appaFeatuers.npy', data_dir + '/appaLabels.npy') ])



