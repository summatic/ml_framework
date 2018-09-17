import os
import tensorflow as tf
from utils import get_logger


class Inferrer:

    def __init__(self, hparams, model, data_manager, sess):
        self.hparams = hparams
        self.model = model
        self.sess = sess
        self.data_manager = data_manager
        self._initialize()

    def _initialize(self):
        self.data_manager.build_placeholders()
        self.model.build_graph(self.data_manager.placeholders)
        self.model.build_inference()

        self.logger = get_logger('inferrer', os.path.join(self.hparams.base_dir, 'inferrer_log.log'))
        self.summary_writer = tf.summary.FileWriter(self.hparams.save_dir)
        self.saver = tf.train.Saver(tf.global_variables())
        self.saver.restore(self.sess, self.hparams.checkpoint_path)

    def infer(self, **kwargs):
        result = self._infer(**kwargs)
        return result

    def _infer(self, **kwargs):
        raise NotImplementedError
