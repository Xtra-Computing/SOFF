from oarf.datasets.fl_dataset import RawDataset


class hit:
    root = 'data/src/chinese_char.hit'
    url = 'http://www.iapr-tc11.org/dataset/OR3C_DAS2010/v1.1/OR3C/offline/character.rar'
    src = 'character.rar'
    src_md5 =
    size =

    file_base = 'sst2'
    shuffle_seed = 0xdeadbeef
    file_md5 =


class casia:
    root = 'data/src/chinese_char.casia'
    url =
    src =
    src_md5 =
    size =

    file_base = 'sst2'
    shuffle_seed = 0xdeadbeef
    file_md5 =


class ChineseCharacterDataset(RawDataset):
    def get_data(self, sample):
        return torch.Tensor(sample[0])

    def get_label(self, sample):
        return float(sample[1])


class HIT(ChineseCharacterDataset):
    def __init__(self):
        pass

    def load_train_dataset(self, _):
        save_path = pathlib.Path(imdb.root).joinpath(
            imdb.file_base + '_{}.pkl'.format(imdb.shuffle_seed)).as_posix()
        with open(save_path, 'rb') as f:
            return pickle.load(f)['train']

    def load_eval_dataset(self, seed):
        save_path = pathlib.Path(imdb.root).joinpath(
            imdb.file_base + '_{}.pkl'.format(imdb.shuffle_seed)).as_posix()
        with open(save_path, 'rb') as f:
            return pickle.load(f)['test']

    def load_test_dataset(self, seed):
        save_path = pathlib.Path(imdb.root).joinpath(
            imdb.file_base + '_{}.pkl'.format(imdb.shuffle_seed)).as_posix()
        with open(save_path, 'rb') as f:
            return pickle.load(f)['val']


class CASIA(ChineseCharacterDataset):
    pass
