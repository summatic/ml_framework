import tensorflow as tf
from models.base_model import BaseModel
from ingredients import create_bidirectional_dynamic_rnn, create_max_pooling1d, cosine_similarity, remove_zero_vectors


class RetrievalBasedDialogModel(BaseModel):

    def __init__(self, hparams, sess, embedding):
        super().__init__(hparams, sess, embedding)

        self.__score = None

    @staticmethod
    def build_embedding_lookup(inputs, embedding):
        embed = tf.nn.embedding_lookup(params=embedding,
                                       ids=inputs)
        return embed

    @staticmethod
    def build_sentence_embedding(scope,
                                 embedded_inputs, input_sequence_lengths, cell_typ, num_layers, num_units,
                                 reuse, keep_prob, pool_size, strides, padding, data_format, name):
        """
        :return: tuple (RNN output, RNN state)
        """
        with tf.variable_scope(scope):
            outputs, state = create_bidirectional_dynamic_rnn(inputs=embedded_inputs,
                                                              sequence_lengths=input_sequence_lengths,
                                                              cell_typ=cell_typ,
                                                              num_layers=num_layers,
                                                              num_units=num_units,
                                                              reuse=reuse,
                                                              keep_prob=keep_prob,
                                                              scope=scope)

            outputs = tf.unstack(tf.concat(outputs, 2), axis=0)
            max_pooled = []
            for output in outputs:
                _output = remove_zero_vectors(output)
                max_pooled.append(tf.reduce_max(_output, axis=0))
            max_pooled = tf.stack(max_pooled)
        return max_pooled

        #     pooled = create_max_pooling1d(inputs=tf.concat(outputs, 2),
        #                                   pool_size=pool_size,
        #                                   strides=strides,
        #                                   padding=padding,
        #                                   data_format=data_format,
        #                                   name=name,
        #                                   scope=scope)
        # return tf.squeeze(pooled, axis=1)

    def _build_graph(self, placeholders):
        embed_contexts = self.build_embedding_lookup(placeholders['contexts'], self.embedding)
        embed_responses = self.build_embedding_lookup(placeholders['responses'], self.embedding)
        embed_neg_responses = self.build_embedding_lookup(placeholders['neg_responses'], self.embedding)

        pooled_contexts = self.build_sentence_embedding(
            scope='context',
            embedded_inputs=embed_contexts,
            input_sequence_lengths=placeholders['context_sequence_lengths'],
            cell_typ=self.hparams.cell_typ,
            num_layers=self.hparams.num_layers,
            num_units=self.hparams.num_units,
            reuse=False,
            keep_prob=self.hparams.keep_prob,
            pool_size=self.hparams.input_length,
            strides=self.hparams.strides,
            padding=self.hparams.padding,
            data_format='channels_last',
            name=None)

        pooled_responses = self.build_sentence_embedding(
            scope='response',
            embedded_inputs=embed_responses,
            input_sequence_lengths=placeholders['response_sequence_lengths'],
            cell_typ=self.hparams.cell_typ,
            num_layers=self.hparams.num_layers,
            num_units=self.hparams.num_units,
            reuse=False,
            keep_prob=self.hparams.keep_prob,
            pool_size=self.hparams.input_length,
            strides=self.hparams.strides,
            padding=self.hparams.padding,
            data_format='channels_last',
            name=None)

        pooled_neg_responses = self.build_sentence_embedding(
            scope='response',
            embedded_inputs=embed_neg_responses,
            input_sequence_lengths=placeholders['neg_response_sequence_lengths'],
            cell_typ=self.hparams.cell_typ,
            num_layers=self.hparams.num_layers,
            num_units=self.hparams.num_units,
            reuse=True,
            keep_prob=self.hparams.keep_prob,
            pool_size=self.hparams.input_length,
            strides=self.hparams.strides,
            padding=self.hparams.padding,
            data_format='channels_last',
            name=None)

        graph_outputs = {'pooled_contexts': pooled_contexts,
                         'pooled_responses': pooled_responses,
                         'pooled_neg_responses': pooled_neg_responses}
        return graph_outputs

    def _build_loss(self):
        graph_outputs = self.graph_outputs

        pooled_contexts = graph_outputs['pooled_contexts']
        pooled_responses = graph_outputs['pooled_responses']
        pooled_neg_responses = graph_outputs['pooled_neg_responses']

        positive_scores = cosine_similarity(pooled_contexts, pooled_responses)
        negative_scores = cosine_similarity(pooled_contexts, pooled_neg_responses)

        ranking_loss = tf.reduce_mean(
            tf.maximum(0.0, 1.0 - positive_scores + negative_scores))

        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name])
        l2_loss *= self.hparams.l2_loss

        mean_pos_score, std_pos_score = tf.nn.moments(positive_scores, axes=[0])
        mean_neg_score, std_neg_score = tf.nn.moments(negative_scores, axes=[0])

        tf.summary.scalar('train/mean_positive_scores', tf.reduce_mean(mean_pos_score))
        tf.summary.scalar('train/mean_negative_scores', tf.reduce_mean(mean_neg_score))
        tf.summary.scalar('train/stddev_positive_scores', tf.reduce_mean(std_pos_score))
        tf.summary.scalar('train/stddev_negative_scores', tf.reduce_mean(std_neg_score))
        tf.summary.scalar('train/ranking_loss', ranking_loss)
        tf.summary.scalar('train/l2_loss', l2_loss)

        return ranking_loss + l2_loss

    def _build_optimizer(self):
        loss = self.loss

        if self.hparams.optimizer == 'adam':
            optimizer = self._create_adam_optimizer()

        # refactoring
        grads, vars = zip(*optimizer.compute_gradients(
            loss, tf.trainable_variables()))
        clipped_gradients, grad_norm = tf.clip_by_global_norm(
            grads, self.hparams.max_grad_norm)
        op = optimizer.apply_gradients(
            zip(clipped_gradients, vars), global_step=self.global_step)
        tf.summary.scalar('grad/gradient_norm', grad_norm)

        return op

    def _output(self):
        return_dict = {'score': self.score}
        return return_dict

    def _build_inference(self):
        graph_outputs = self.graph_outputs
        pooled_contexts = graph_outputs['pooled_contexts']
        pooled_responses = graph_outputs['pooled_responses']
        positive_scores = cosine_similarity(pooled_contexts, pooled_responses)

        self.__score = positive_scores

        mean_pos_score, std_pos_score = tf.nn.moments(positive_scores, axes=[0])

        tf.summary.scalar('val/mean_positive_scores', mean_pos_score)
        tf.summary.scalar('val/stddev_positive_scores', std_pos_score)
        return_dict = {'score': positive_scores, 'pooled_contexts': graph_outputs['pooled_contexts']}

        return return_dict

    @property
    def score(self):
        return self.__score
