import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math

# Model
class VGG(nn.Module):
    def __init__(self, features, num_classes, batch_per_lot=None, sigma=None):
        super(VGG, self).__init__()
        self.features = features
        
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024, momentum=0.66),
            nn.Linear(1024, 256),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256, momentum=0.66),
            nn.Linear(256, num_classes),
            nn.LogSoftmax(dim=1)
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
def conv_unit(input, output, mp=False):
    if mp:
        return [nn.Conv2d(input, output, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(), 
               nn.BatchNorm2d(output, momentum=0.66), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
    else:
        return [nn.Conv2d(input, output, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(), 
               nn.BatchNorm2d(output, momentum=0.66)]

def make_layers():
    layers = []
    layers += [nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1), nn.LeakyReLU(), 
               nn.BatchNorm2d(64, momentum=0.66)]

    layers += conv_unit(64, 128)
    layers += conv_unit(128, 128, mp=True)

    layers += conv_unit(128, 256)
    layers += conv_unit(256, 256, mp=True)

    layers += conv_unit(256, 384)
    layers += conv_unit(384, 384)
    layers += conv_unit(384, 384, mp=True)

    layers += conv_unit(384, 512)
    layers += conv_unit(512, 512)
    layers += conv_unit(512, 512, mp=True)

    layers += [nn.Flatten()]

    return nn.Sequential(*layers)

def get_model(args):
    model = VGG(make_layers(), 3755)
    return model

def get_loss_func(args):
    return F.nll_loss

def get_metric_func(args):
    def metric_func(output, target):
        pred = output.argmax(dim=1, keepdim=True)    # get the index of the max log-probability
        correct = pred.eq(target.view_as(pred)).sum().item()
        return correct
    return metric_func
