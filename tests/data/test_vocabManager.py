import os
from unittest import TestCase
from data.vocab_manager import VocabManager
from hparams import load_hparams

abspath = os.path.abspath(os.path.dirname(__file__))
hparams = load_hparams(os.path.join(abspath, '../data/test_files/datasets/unlabeled_pair/'))
temp_path = 'temp'


class TestVocabManager(TestCase):
    def setUp(self):
        self.vocab_manager = VocabManager(hparams=hparams)

    def tearDown(self):
        del self.vocab_manager

    def test_build_vocab(self):
        with open('test_files/sentences.txt', 'r') as f:
            sentences = f.readlines()

        answer = {'1': 2, '2': 2, '3': 2, '4': 1, '5': 2, '6': 1}

        self.vocab_manager.build_vocab(sentences)
        self.assertDictEqual(answer, self.vocab_manager.vocab_counts)

    def test_save_vocab_counts(self):
        self.vocab_manager.vocab_counts = {'1': 2, '2': 2, '3': 2, '4': 1, '5': 2, '6': 1}
        self.vocab_manager.save_vocab_counts()

        isfile = os.path.isfile(self.vocab_manager.vocab_params.vocab_count_path)
        self.assertTrue(isfile)

    def test_load_vocab_counts(self):
        answer = {'1': 2, '2': 2, '3': 2, '5': 2}

        self.vocab_manager.load_vocab_counts()
        self.assertDictEqual(answer, self.vocab_manager.vocab_counts)

    def test_index_vocabs(self):
        answer = {'UNK': 0, 'PAD': 1, 'SOS': 2, 'EOS': 3,
                  '1': 4, '2': 5, '3': 6, '5': 7}

        self.vocab_manager.index_vocabs()
        self.assertDictEqual(answer, self.vocab_manager.vocab_index)

    def test_save_vocab_index(self):
        self.vocab_manager.index_vocabs()
        self.vocab_manager.save_vocab_index()

        isfile = os.path.isfile(self.vocab_manager.vocab_params.vocab_index_path)
        self.assertTrue(isfile)

    def test_load_vocab_index(self):
        answer = {'UNK': 0, 'PAD': 1, 'SOS': 2, 'EOS': 3,
                  '1': 4, '2': 5, '3': 6, '5': 7}

        self.vocab_manager.load_vocab_index()
        self.assertDictEqual(answer, self.vocab_manager.vocab_index)


if __name__ == '__main__':
    testcase = TestCase()
    testcase.run()
