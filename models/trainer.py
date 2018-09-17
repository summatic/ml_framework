import os
from time import time
import tensorflow as tf
from utils import get_logger


class Trainer:

    def __init__(self, hparams, model, data_manager, sess):
        self.hparams = hparams
        self.model = model
        self.sess = sess
        self.data_manager = data_manager
        self._initialize()

    def _initialize(self):
        self.data_manager.read_files()
        self.data_manager.build_placeholders()
        self.model.build_graph(self.data_manager.placeholders)
        self.model.build_loss()
        self.model.build_inference()
        self.model.build_optimizer()

        self.logger = get_logger('trainer', os.path.join(self.hparams.base_dir, 'trainer_log.log'))
        self.summary_writer = tf.summary.FileWriter(self.hparams.save_dir)
        self.saver = tf.train.Saver(tf.global_variables())

    def _log(self, loss, prev_time, epoch, global_step):
        raise NotImplementedError

    def _summary(self, summary, global_step):
        self.summary_writer.add_summary(summary, global_step)
        self.summary_writer.flush()

    def _checkpoint(self, global_step):
        self.saver.save(self.sess, os.path.join(self.hparams.save_dir, 'model.ckpt'), global_step=global_step)

    def _val(self, global_step, tol):
        raise NotImplementedError

    def train(self):
        model_output = self.model.output
        self.sess.run(tf.global_variables_initializer())
        global_step = self.sess.run(self.model.global_step)

        prev_time = time()
        early_stopping_metric = 0
        best_ckpt = 0
        early_stopping_steps = 5
        tolerance = 0

        self.logger.warning('Training Start')
        for epoch in range(self.hparams.epochs):
            data_gen = self.data_manager.make_data_generator(mode='train')

            for data in data_gen:
                if tolerance >= early_stopping_steps:
                    break

                global_step += 1
                result = self.sess.run(model_output, feed_dict=data)

                if global_step % self.hparams.log_steps == 0:
                    self._log(loss=result['loss'], prev_time=prev_time, epoch=epoch, global_step=global_step)
                    prev_time = time()

                if global_step % self.hparams.summary_steps == 0:
                    self._summary(summary=result['summary'], global_step=global_step)

                if global_step % self.hparams.val_steps == 0:
                    self._checkpoint(global_step=global_step)
                    metric = self._val(global_step=global_step, tol=tolerance)

                    if metric > early_stopping_metric:
                        early_stopping_metric = metric
                        best_ckpt = global_step
                        tolerance = 0
                    else:
                        tolerance += 1
            if tolerance >= early_stopping_steps:
                break

        self.logger.warning('Best Metric: {:.5f} at step {}'.format(early_stopping_metric, best_ckpt))
        self.logger.warning('Training End')
