import tensorflow as tf
from models.seq2seq.model import Seq2seqModel


class TestSeq2seqModel(tf.test.TestCase):
    def setUp(self):
        hparams = None
        sess = self.test_session()
        embedding = None
        self.model = Seq2seqModel(hparams, sess, embedding)

    def test_build_placeholders(self):
        results = self.model.build_placeholders()

        self.skipTest('')

    def test_build_encoding_layer(self):

        self.skipTest('')

    def test_build_decoding_layer(self):

        self.skipTest('')

if __name__ == '__main__':
    testcase = TestSeq2seqModel()
    testcase.run()
