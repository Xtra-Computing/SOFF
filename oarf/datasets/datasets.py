import re
from typing import Callable, List
from oarf.metrics.accuracy import binary, top1, top5
from oarf.datasets import cifar10, sentiment, face_age
from oarf.datasets.fl_dataset import (
    IidDataset, DirichletQuantitySkewDataset,
    PowerLawQuantitySkewDataset, DirichletLabelSkewDataset, RealisticDataset)

data_splitting_name_to_class = {
    'iid': IidDataset,
    'quantity-skew-dirichlet': DirichletQuantitySkewDataset,
    'quantity-skew-powerlaw': PowerLawQuantitySkewDataset,
    'label-skew-dirichlet': DirichletLabelSkewDataset,
    'realistic': RealisticDataset,
}

dataset_name_to_class = {
    'cifar10': cifar10.CIFAR10,
    'sentiment:imdb': sentiment.IMDB,
    'sentiment:sst2': sentiment.SST2,
    'sentiment:amazon': sentiment.Amazon,
    'face:allagefaces': face_age.AllAgeFaces,
    'face:appa': face_age.AppaRealAge,
    'face:wiki': face_age.Wiki,
    'face:utk': face_age.UTKFaces,
    # 'alphanum:fnt',
    # 'alphanum:hnd',
    # 'chinese:hit',
    # 'chinese:casia'
    # 'trffic:la',
    # 'traffic:bay'
}


def create_dataset(datasets: List[str], data_splitting: str, *args, **kwargs):
    return data_splitting_name_to_class[data_splitting](
        *args, **kwargs,
        datasets=[dataset_name_to_class[d](
            *args, **kwargs) for d in datasets])


additional_criterion_name_map = {
    'cifar10': [top1, top5],
    'sentiment': [binary],
    'face': [face_age.mae, face_age.acc, face_age.one_off_acc],
}


def init_additional_criteria(name: str) -> List[Callable]:
    return additional_criterion_name_map[re.sub(r':.*$', '', name).lower()]
