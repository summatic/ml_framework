from random import shuffle, seed, sample
import tensorflow as tf
from tqdm import tqdm
from data.vocab_manager import VocabManager

seed(910417)


class DataManager:

    def __init__(self, hparams):
        self.hparams = hparams

        self.vocab_manager = VocabManager(hparams=hparams)
        self.vocab_manager.load_vocab_index()
        self.vocab_index = self.vocab_manager.vocab_index.copy()

        self.dataset = None
        self.__placeholders = None

    def indexing(self, sentence):
        indexed = [self.vocab_index['SOS']]

        for vocab in sentence.split():
            try:
                indexed.append(self.vocab_index[vocab])
            except KeyError:
                indexed.append(self.vocab_index['UNK'])

        pad_len = self.hparams.input_length - len(indexed) - 1
        indexed += [self.vocab_index['PAD']] * pad_len
        indexed += [self.vocab_index['EOS']]

        return indexed

    def _read_file(self, file_path, sep='\t'):
        """"""
        raise NotImplementedError

    def read_files(self):
        dataset = {
            'train': self._read_file(self.hparams.train_path),
            # 'test': self._read_file(self.hparams.test_path),
            'val': self._read_file(self.hparams.val_path)
        }

        self.dataset = dataset

    def build_placeholders(self):
        self.__placeholders = self._build_placeholders()

    def _build_placeholders(self):
        raise NotImplementedError

    @property
    def placeholders(self):
        return self.__placeholders

    @property
    def vocab_size(self):
        return len(self.vocab_index)

    def make_data_generator(self, mode):
        raise NotImplementedError

    def make_instance(self, **kwargs):
        raise NotImplementedError


class LabeledPairDataManager(DataManager):

    def __init__(self, hparams, sess):
        super().__init__(hparams=hparams, sess=sess)

    def _read_file(self, file_path, sep='\t'):
        data = []
        f = open(file_path, 'r')
        for line in tqdm(f):
            s0, s1, label = line.strip().split(sep)
            s0_length = len(s0.split())
            s1_length = len(s1.split())
            s0 = self.indexing(s0)
            s1 = self.indexing(s1)
            label = int(label)
            data.append((s0, s1, s0_length, s1_length, label))
        f.close()

        return data

    def _build_placeholders(self):
        sent0 = tf.placeholder(tf.int32, [None, None], name='sent0')
        sent1 = tf.placeholder(tf.int32, [None, None], name='sent1')

        sent0_sequence_lengths = tf.placeholder(tf.int32, [None], name='sent0_sequence_lengths')
        sent1_sequence_lengths = tf.placeholder(tf.int32, [None], name='response_sequence_lengths')

        max_sent0_length = tf.reduce_max(sent0_sequence_lengths)
        max_sent1_length = tf.reduce_max(sent1_sequence_lengths)

        label = tf.placeholder(tf.int32, [None], name='label')

        return_dict = {
            'sent0': sent0,
            'sent1': sent1,
            'sent0_sequence_lengths': sent0_sequence_lengths,
            'response_sequence_lengths': sent1_sequence_lengths,
            'max_sent0_length': max_sent0_length,
            'max_sent1_length': max_sent1_length,
            'label': label
        }

        return return_dict

    def make_data_generator(self, mode):
        placeholders = self.placeholders

        data = self.dataset[mode]
        if mode == 'train':
            shuffle(data)

        batch_size = self.hparams.batch_size
        for i in range(0, len(data), batch_size):
            _data = {k: [] for k, _ in placeholders.items()}
            for (sent0, sent1, sent0_sequence_length, sent1_sequence_length, label) in data[i:i+batch_size]:
                _data['sent0'].append(sent0)
                _data['sent1'].append(sent1)
                _data['sent0_sequence_lengths'].append(sent0_sequence_length)
                _data['sent1_sequence_lengths'].append(sent1_sequence_length)

                if mode == 'train':
                    _data['label'].append(label)

            feed_dict = {placeholders[k]: v for k, v in _data.items()}

            yield feed_dict

    def make_instance(self, **kwargs):
        sent0 = kwargs['sent0']
        sent1 = kwargs['sent1']

        return {'sent0': self.indexing(sent0), 'sent1': self.indexing(sent1)}


class UnlabeledPairDataManager(DataManager):

    def __init__(self, hparams, sess):
        super().__init__(hparams=hparams, sess=sess)

    def _read_file(self, file_path, sep='\t'):
        data = []
        f = open(file_path, 'r')
        for line in tqdm(f):
            s0, s1 = line.strip().split(sep)
            s0_length = len(s0.split())
            s1_length = len(s1.split())
            s0 = self.indexing(s0)
            s1 = self.indexing(s1)
            data.append((s0, s1, s0_length, s1_length))
        f.close()

        return data

    def _build_placeholders(self):
        sent0 = tf.placeholder(tf.int32, [None, None], name='sent0')
        sent1 = tf.placeholder(tf.int32, [None, None], name='sent1')

        sent0_sequence_lengths = tf.placeholder(tf.int32, [None], name='sent0_sequence_lengths')
        sent1_sequence_lengths = tf.placeholder(tf.int32, [None], name='response_sequence_lengths')

        max_sent0_length = tf.reduce_max(sent0_sequence_lengths)
        max_sent1_length = tf.reduce_max(sent1_sequence_lengths)

        return_dict = {
            'sent0': sent0,
            'sent1': sent1,
            'sent0_sequence_lengths': sent0_sequence_lengths,
            'sent1_sequence_lengths': sent1_sequence_lengths,
            'max_sent0_length': max_sent0_length,
            'max_sent1_length': max_sent1_length
        }

        return return_dict

    def make_data_generator(self, mode):
        placeholders = self.placeholders

        data = self.dataset[mode]
        if mode == 'train':
            shuffle(data)

        batch_size = self.hparams.batch_size
        for i in range(0, len(data), batch_size):
            _data = {k: [] for k, _ in placeholders.items()}
            for (sent0, sent1, sent0_sequence_length, sent1_sequence_length, label) in data[i:i+batch_size]:
                _data['sent0'].append(sent0)
                _data['sent1'].append(sent1)
                _data['sent0_sequence_lengths'].append(sent0_sequence_length)
                _data['sent1_sequence_lengths'].append(sent1_sequence_length)
            feed_dict = {placeholders[k]: v for k, v in _data.items()}

            yield feed_dict

    def make_instance(self, **kwargs):
        sent0 = kwargs['sent0']
        sent1 = kwargs['sent1']

        return {'sent0': self.indexing(sent0), 'sent1': self.indexing(sent1)}
