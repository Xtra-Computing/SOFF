from torch import nn
from torchvision.models import vgg16


class VGG16(nn.Module):
    def __init__(self, num_classes, pretrained=True, *_, **__):
        super().__init__()
        self.vgg16 = vgg16(pretrained=pretrained, progress=True)
        self.vgg16.classifier[6] = nn.Linear(4096, num_classes)
        self.forward = self.vgg16.forward
