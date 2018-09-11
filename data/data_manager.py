import abc
import tensorflow as tf
from data.vocab_manager import VocabManager


class DataManager(abc.ABC):

    def __init__(self, hparams):
        self.hparams = hparams

        data_params = self.hparams['data']
        train_file_path = data_params['train_file_path']
        test_file_path = data_params['test_file_path']
        val_file_path = data_params['val_file_path']

        self.dataset = {
            'train': tf.data.TextLineDataset(filenames=train_file_path),
            'test': tf.data.TextLineDataset(filenames=test_file_path),
            'val': tf.data.TextLineDataset(filenames=val_file_path)
        }

        self.vocab_manager = VocabManager(hparams=hparams)
        self.vocab_manager.load_vocab_index()
        self.vocab_index = self.vocab_manager.vocab_index.copy()

    @abc.abstractmethod
    def _readlines(self, lines, sep):
        """"""

    def indexing(self, sentences):
        indexed = [self.vocab_index['SOS']]
        for vocab in sentences.split():
            try:
                indexed.append(self.vocab_index[vocab])
            except KeyError:
                indexed.append(self.vocab_index['UNK'])

        pad_len = self.hparams['model']['input_length'] - len(indexed) - 1
        indexed += [self.vocab_index['PAD']] * pad_len
        indexed += [self.vocab_index['EOS']]

        return indexed

    def apply(self):
        for _, dataset in self.dataset.items():
            dataset.map(self._readlines)


class LabeledPairDataManager(DataManager):

    def __init__(self, hparams):
        super().__init__(hparams=hparams)

    def _readlines(self, lines, sep):
        s0, s1, label = lines.split(sep)
        return s0, s1, label
