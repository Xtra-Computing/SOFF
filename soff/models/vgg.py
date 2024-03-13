"""VGG models (wrapper of pytorch's models)"""
from torchvision.models import vgg11, vgg13, vgg16, vgg19, VGG
from ..utils.arg_parser import ArgParseOption, options, require


@options(
    "VGG Model Configs",
    ArgParseOption(
        'vgg.p', 'vgg.pretrained', action='store_true',
        help="Load pretrained VGG model parameters."))
@require('model.vgg.pretrained')
class _VGG(VGG):
    def __init__(self, cfg, dataset, vgg_creator) -> None:
        model = vgg_creator(
            pretrained=cfg.model.vgg.pretrained,
            num_classes=dataset.num_classes(), progress=True)
        self.__dict__ = model.__dict__


class VGG11(_VGG):
    def __init__(self, cfg, dataset) -> None:
        super().__init__(cfg, dataset, vgg_creator=vgg11)


class VGG13(_VGG):
    def __init__(self, cfg, dataset) -> None:
        super().__init__(cfg, dataset, vgg_creator=vgg13)


class VGG16(_VGG):
    def __init__(self, cfg, dataset) -> None:
        super().__init__(cfg, dataset, vgg_creator=vgg16)


class VGG19(_VGG):
    def __init__(self, cfg, dataset) -> None:
        super().__init__(cfg, dataset, vgg_creator=vgg19)
