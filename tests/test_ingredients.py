import os
import numpy as np
import tensorflow as tf
from ingredients import create_rnn_cell, create_bidirectional_dynamic_rnn, create_embedding, \
    remove_zero_vectors, create_max_pooling1d, cosine_similarity
from hparams import load_hparams
abspath = os.path.abspath(os.path.dirname(__file__))
hparams = load_hparams(os.path.join(abspath, './data/test_files/datasets/unlabeled_pair/'))


class TestIngredients(tf.test.TestCase):

    def test_create_rnn_cell(self):
        cell_typ = hparams.cell_typ
        num_layers = hparams.num_layers
        num_units = hparams.num_units
        reuse = True
        keep_prob = hparams.keep_prob
        results = create_rnn_cell(cell_typ=cell_typ,
                                  num_layers=num_layers,
                                  num_units=num_units,
                                  reuse=reuse,
                                  keep_prob=keep_prob)
        self.skipTest('')

    def test_create_bidirectional_dynamic_rnn(self):  # TODO: self.assertShapeEqual(np, tensor)
        batch_size = hparams.batch_size
        input_length = hparams.input_length

        cell_typ = hparams.cell_typ
        num_layers = hparams.num_layers
        num_units = hparams.num_units
        reuse = False
        keep_prob = hparams.keep_prob
        scope = 'test'

        # tensors
        inputs = tf.placeholder(shape=[batch_size, input_length, num_units], dtype=tf.float32)
        sequence_lengths = tf.placeholder(shape=[batch_size], dtype=tf.int32)

        # numpy
        answer_output = np.random.rand(batch_size, input_length, num_units)
        answer_state = np.random.rand(batch_size, num_units)

        outputs_op, state_op = create_bidirectional_dynamic_rnn(
            inputs=inputs,
            sequence_lengths=sequence_lengths,
            cell_typ=cell_typ,
            num_layers=num_layers,
            num_units=num_units,
            reuse=reuse,
            keep_prob=keep_prob,
            scope=scope)

        self.assertShapeEqual(np_array=answer_output, tf_tensor=outputs_op[0])
        self.assertShapeEqual(np_array=answer_output, tf_tensor=outputs_op[1])
        self.assertShapeEqual(np_array=answer_state, tf_tensor=state_op[0][0])
        self.assertShapeEqual(np_array=answer_state, tf_tensor=state_op[1][0])

    def test_create_embedding(self):
        pretrained_path = None
        name = 'test'
        vocab_size = 100
        embedding_size = 20
        trainable = True

        answer = np.random.random(size=(vocab_size, embedding_size))
        ops = create_embedding(pretrained_path=pretrained_path,
                               name=name,
                               vocab_size=vocab_size,
                               embedding_size=embedding_size,
                               trainable=trainable)
        self.assertTupleEqual(answer.shape, tuple(ops.shape))

    def test_remove_zero_vectors(self):
        time_step = 10
        embedding_size = 100
        answer_nonzero = np.random.randn(time_step, embedding_size)
        answer_zero = np.zeros(shape=(3, embedding_size))
        answer = np.concatenate((answer_nonzero, answer_zero), axis=0)

        with self.test_session() as sess:
            inputs = tf.placeholder(shape=[time_step + 3, embedding_size], dtype=tf.float32)
            ops = remove_zero_vectors(inputs)
            results = sess.run(ops, feed_dict={inputs: answer})
        self.assertNDArrayNear(answer_nonzero, results, 1e-5)

    def test_create_cosine_similarity(self):
        def cosine(m1, m2):
            m1 = m1 / np.linalg.norm(m1, axis=1).reshape(-1, 1)
            m2 = m2 / np.linalg.norm(m2, axis=1).reshape(-1, 1)
            return np.sum(np.multiply(m1, m2), axis=1)
        input1 = tf.placeholder(shape=[2, 10], dtype=tf.float32)
        input2 = tf.placeholder(shape=[2, 10], dtype=tf.float32)

        array1 = np.random.randn(2, 10)
        array2 = np.random.randn(2, 10)

        answer = cosine(array1, array2)

        with self.test_session() as sess:
            cossim = cosine_similarity(input1, input2)
            result = sess.run(cossim, feed_dict={input1: array1, input2: array2})
            self.assertNDArrayNear(answer, result, 1e-5)

if __name__ == '__main__':
    testcase = tf.test.TestCase()
    testcase.run()
