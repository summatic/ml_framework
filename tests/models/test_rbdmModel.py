import os
import numpy as np
import tensorflow as tf
from models.rbdm.model import RetrievalBasedDialogModel
from models.rbdm.data_manager import RBDMDataManager
from ingredients import create_embedding
from hparams import load_hparams

abspath = os.path.abspath(os.path.dirname(__file__))
hparams = load_hparams(os.path.join(abspath, '../data/test_files/datasets/unlabeled_pair/'))
data_manager = RBDMDataManager(hparams)
embedding = create_embedding(pretrained_path=None,
                             name='embedding',
                             vocab_size=data_manager.vocab_size,
                             embedding_size=hparams.embedding_size)


class TestRetrievalBasedDialogModel(tf.test.TestCase):
    def setUp(self):
        sess = self.test_session()
        self.data_manager = data_manager
        self.model = RetrievalBasedDialogModel(hparams, sess, embedding)
        self.data_manager.build_placeholders()

    def tearDown(self):
        del self.model

    # def test_build_embedding_lookup(self):
    #     batch_size = hparams.batch_size
    #     input_length = hparams.input_length
    #     embedding_size = hparams.embedding_size
    #
    #     # tensors
    #     inputs = tf.placeholder(tf.int32, [batch_size, input_length], name='test_inputs')
    #     _embedding = self.model.embedding
    #
    #     # numpy
    #     answer = np.random.rand(batch_size, input_length, embedding_size)
    #
    #     ops = self.model.build_embedding_lookup(inputs, _embedding)
    #     self.assertShapeEqual(np_array=answer, tf_tensor=ops)
    #
    # def test_build_sentence_embedding(self):
    #     batch_size = hparams.batch_size
    #     input_length = hparams.input_length
    #
    #     scope = 'test'
    #     cell_typ = hparams.cell_typ
    #     num_layers = hparams.num_layers
    #     num_units = hparams.num_units
    #     keep_prob = hparams.keep_prob
    #     pool_size = hparams.input_length
    #     strides = hparams.strides
    #     padding = hparams.padding
    #     data_format = hparams.data_format
    #
    #     name = None
    #
    #     # tensors
    #     inputs = tf.placeholder(tf.int32, [batch_size, input_length], name='test_inputs')
    #     sequence_lengths = tf.placeholder(tf.int32, [batch_size], name='test_sequence_lengths')
    #
    #     # numpy
    #     answer = np.random.rand(batch_size, 2*num_units)
    #
    #     embed_inputs = self.model.build_embedding_lookup(inputs, self.model.embedding)
    #     ops = self.model.build_sentence_embedding(scope=scope, embedded_inputs=embed_inputs,
    #                                               input_sequence_lengths=sequence_lengths, cell_typ=cell_typ,
    #                                               num_layers=num_layers, num_units=num_units,
    #                                               keep_prob=keep_prob, pool_size=pool_size, strides=strides,
    #                                               padding=padding, data_format=data_format, name=name)
    #
    #     self.assertShapeEqual(np_array=answer, tf_tensor=ops)

    # def test_build_graph(self):
    #     placeholders = self.data_manager.placeholders
    #
    #     self.model.build_graph(placeholders)
    #
    #     self.skipTest('')
    #
    # def test_build_loss(self):  # TODO: reset graph
    #     self.data_manager.build_placeholders()
    #     placeholders = self.data_manager.placeholders
    #     self.model.build_graph(placeholders)
    #
    #     self.model.build_loss()
    #
    #     self.skipTest('')
    #
    # def test_build_optimizer(self):    # TODO: reset graph
    #     placeholders = self.data_manager.placeholders
    #     self.model.build_graph(placeholders)
    #     self.model.build_loss()
    #
    #     self.model.build_optimizer()
    #
    #     self.skipTest('')

    def test_build_inference(self):
        placeholders = self.data_manager.placeholders
        self.model.build_graph(placeholders)
        self.model.build_inference()

        self.skipTest('')


if __name__ == '__main__':
    testcase = TestRetrievalBasedDialogModel()
    testcase.run()
