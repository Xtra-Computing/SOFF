import argparse
from oarf.utils.logging import init_logging
from oarf.datasets.datasets import dataset_name_to_class

init_logging()
parser = argparse.ArgumentParser()
parser.add_argument('-ds', '--datasets', choices=dataset_name_to_class.keys())
args = parser.parse_args()

dataset_name_to_class[args.datasets]()
