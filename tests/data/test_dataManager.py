import os
from unittest import TestCase
from data.data_manager import DataManager, LabeledPairDataManager
from hparams import load_hparams

abspath = os.path.abspath(os.path.dirname(__file__))
hparams = load_hparams(os.path.join(abspath, '../data/test_files/datasets/unlabeled_pair/'))


class TestDataManager(TestCase):
    def setUp(self):
        self.sess = None
        self.data_manager = DataManager(hparams=hparams)

    def tearDown(self):
        del self.data_manager

    def test_indexing(self):
        sentence = '1 4 6 8 2 3'
        answer = [2, 4, 0, 0, 0, 5, 6, 1, 1, 3]

        result = self.data_manager.indexing(sentence)

        self.assertListEqual(answer, result)


class TestLabeledPairDataManager(TestCase):
    def setUp(self):
        self.sess = None
        self.data_manager = LabeledPairDataManager(hparams=hparams, sess=self.sess)

    def tearDown(self):
        del self.data_manager

    def test__read_file(self):
        answers = [([2, 0, 0, 0, 0, 1, 1, 1, 1, 3], [2, 0, 6, 5, 4, 1, 1, 1, 1, 3], 0),
                   ([2, 0, 6, 5, 4, 1, 1, 1, 1, 3], [2, 4, 0, 4, 5, 1, 1, 1, 1, 3], 1)]
        results = self.data_manager._read_file(self.data_manager.hparams['data']['test'])

        for idx, (answer, result) in enumerate(zip(answers, results)):
            with self.subTest(idx=idx):
                self.assertTupleEqual(answer, result)

    def test_make_instance(self):
        sent0 = '1 4 6 8 2 3'
        sent1 = '1 4 6 8 2 3'

        answer = {'sent0': [2, 4, 0, 0, 0, 5, 6, 1, 1, 3],
                  'sent1': [2, 4, 0, 0, 0, 5, 6, 1, 1, 3]}
        result = self.data_manager.make_instance(sent0=sent0, sent1=sent1)
        self.assertDictEqual(answer, result)


if __name__ == '__main__':
    testcase = TestCase()
    testcase.run()
