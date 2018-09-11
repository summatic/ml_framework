from unittest import TestCase
from data.data_manager import DataManager, LabeledPairDataManager

hparams = {
    'vocabs': {
        'count_file_path': 'test_files/temp_count_file.txt',
        'index_file_path': 'test_files/temp_index_file.txt',
        'min_count': 2},
    'data': {
        'train_file_path': 'test/files/datasets/train.txt',
        'test_file_path': 'test/files/datasets/test.txt',
        'val_file_path': 'test/files/datasets/val.txt'
    },
    'model': {
        'input_length': 10
    }
}


class TestDataManager(TestCase):
    def setUp(self):
        self.data_manager = DataManager(hparams)

    def tearDown(self):
        del self.data_manager

    def test_indexing(self):
        sentences = '1 4 6 8 2 3'
        answer = [2, 4, 0, 0, 0, 5, 6, 1, 1, 3]

        result = self.data_manager.indexing(sentences)
        self.assertListEqual(answer, result)


class TestLabeledPairDataManager(TestCase):
    def setUp(self):
        self.data_manager = LabeledPairDataManager(hparams)

    def tearDown(self):
        del self.data_manager

    def test_apply(self):


if __name__ == '__main__':
    testcase = TestCase()
    testcase.run()
