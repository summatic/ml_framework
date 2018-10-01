from time import time
import numpy as np
from models.trainer import Trainer


class RBDMTrainer(Trainer):

    def __init__(self, hparams, model, data_manager, sess):
        super().__init__(hparams, model, data_manager, sess)

    def _log(self, loss, prev_time, epoch, global_step):
        now_time = time()
        instances_per_second = self.hparams.log_steps * self.hparams.batch_size / (now_time - prev_time)

        self.logger.warning(
            '[Epoch {} | Step {}] Loss: {:.5f}, Speed: {:.2f} instances/sec'.format(
                epoch, global_step, loss, instances_per_second))

    def _val(self, global_step, tol):
        score_op = self.model.inference['score']
        data_gen = self.data_manager.make_data_generator(mode='val')

        scores = []
        for data in data_gen:
            score = self.sess.run(score_op, feed_dict=data)
            scores.extend(score)
        score = np.mean(np.array(scores))

        self.logger.warning('================Validation results================')
        self.logger.warning('step {} / tolerance {}'.format(global_step, tol))
        self.logger.warning('Score: {:.5f}'.format(score))
        self.logger.warning('==================================================')

        return score
