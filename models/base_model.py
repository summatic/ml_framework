import tensorflow as tf


class BaseModel:

    def __init__(self, hparams, sess, embedding):
        self.hparams = hparams
        self.sess = sess
        self.embedding = embedding

        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.__placeholders = None
        self.__graph_outputs = None
        self.__loss = None
        self.__inference = None
        self.__optimizer = None
        self.__summarizer = None

    # placeholder
    @property
    def placeholders(self):
        return self.__placeholders

    def build_graph(self, placeholder):
        self.__graph_outputs = self._build_graph(placeholder)

    def _build_graph(self, placeholder):
        raise NotImplementedError

    # graph output
    @property
    def graph_outputs(self):
        return self.__graph_outputs

    # loss
    def build_loss(self):
        self.__loss = self._build_loss()

    def _build_loss(self):
        raise NotImplementedError

    @property
    def loss(self):
        return self.__loss

    # inference
    def build_inference(self):
        self.__inference = self._build_inference()

    def _build_inference(self):
        raise NotImplementedError

    @property
    def inference(self):
        return self.__inference

    # optimizer
    def build_optimizer(self):
        self.__optimizer = self._build_optimizer()
        tf.summary.scalar('total_loss', self.loss)
        self.__summarizer = tf.summary.merge_all()

    def _build_optimizer(self):
        raise NotImplementedError

    @property
    def optimizer(self):
        return self.__optimizer

    # summarizer
    @property
    def summarizer(self):
        return self.__summarizer

    # model output
    @property
    def output(self):
        return_dict = {
            'optimizer': self.optimizer,
            'loss': self.loss,
            'summary': self.summarizer
        }

        _output = self._output()
        if _output:
            return_dict.update(_output)

        return return_dict

    def _output(self):
        raise NotImplementedError

    # create optimizer
    def _create_adam_optimizer(self):  # TODO: 분리하기.
        learning_rate = self.hparams.learning_rate
        beta1 = self.hparams.beta1
        beta2 = self.hparams.beta2
        epsilon = self.hparams.epsilon

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                           beta1=beta1, beta2=beta2, epsilon=epsilon)
        return optimizer
