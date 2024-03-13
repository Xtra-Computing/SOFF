from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import json
from ..utils.arg_parser import TrainingConfParser
from ..utils.logging import LogConfParser, init_logging
from ..datasets import DataConfParser, create_dataset
from ..datasets.raw_dataset import RawDataConfParser, create_raw_dataset


class FLDatasetInitializerArgParser(
        DataConfParser, TrainingConfParser, LogConfParser):
    pass


class RawDatasetInitializerArgParser(
        RawDataConfParser, TrainingConfParser, LogConfParser):
    def __init__(self, *args, stag='', ltag='', **kwargs):
        super().__init__(*args, stag=stag, ltag=ltag, **kwargs)
        self.add_argument(
            '-dt.r.si', '--data.raw.split-id', type=int, default=0,
            help="Some datasets (especially desinged for delegated FL split) "
            "requires split-id to initialize")


class DatasetMetaArgParser:
    def __init__(self) -> None:
        self.parser = ArgumentParser(
            prefix_chars='+', formatter_class=ArgumentDefaultsHelpFormatter)
        self.parser.add_argument(
            'cmd', type=str, metavar='CMD', choices=('fl', 'raw'),
            help="Use raw/fl dataset initializer.")


def init_raw_ds(args):
    parser = RawDatasetInitializerArgParser()
    parser.print_help()
    cfg = parser.parse_args()
    init_logging(cfg)

    print(json.dumps(cfg, indent=2))
    for mode in ('train', 'eval', 'test'):
        print(f"Mode: {mode}")
        ds = create_raw_dataset(
            cfg, name=cfg.data.raw.datasets[0], mode=mode,
            split_id=cfg.data.raw.split_id)
        print(len(ds.load_train_descs()))
        print(len(ds.load_eval_descs()))
        print(len(ds.load_test_descs()))


def init_fl_ds(args):
    parser = FLDatasetInitializerArgParser()
    parser.print_help()
    cfg = parser.parse_args()
    init_logging(cfg)

    print(json.dumps(cfg, indent=2))
    for mode in ('train', 'eval', 'test'):
        print(f"Mode: {mode}")
        ds = create_dataset(
            cfg, datasets=[cfg.data.raw.datasets[0]],
            mode=mode, split_id=cfg.data.fl_split.num)
        print(ds[0][0].shape)


def main():
    args, unknown = DatasetMetaArgParser().parser.parse_known_args()
    if args.cmd == 'fl':
        init_fl_ds(unknown)
    elif args.cmd == 'raw':
        init_raw_ds(unknown)
    else:
        raise RuntimeError(f"Unknown command {args.cmd}")


if __name__ == "__main__":
    main()
