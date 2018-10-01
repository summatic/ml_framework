import tensorflow as tf
from models.base_model import BaseModel
from ingredients import create_bidirectional_dynamic_rnn, create_rnn_cell


class Seq2seqModel(BaseModel):

    def __init__(self, hparams, sess, embedding):
        super().__init__(hparams, sess, embedding)

    @staticmethod
    def build_embedding_lookup(inputs, embedding):
        embed = tf.nn.embedding_lookup(param=embedding,
                                       ids=inputs)
        return embed

    @staticmethod
    def build_encoding_layer(embedded_inputs, input_sequence_lengths, cell_typ, num_layers, num_units, keep_prob):
        """
        :return: tuple (RNN output, RNN state)
        """
        with tf.variable_scope('encoding'):
            outputs, state = create_bidirectional_dynamic_rnn(inputs=embedded_inputs,
                                                              sequence_lengths=input_sequence_lengths,
                                                              cell_typ=cell_typ,
                                                              num_layers=num_layers,
                                                              num_units=num_units,
                                                              reuse=False,
                                                              keep_prob=keep_prob,
                                                              scope='encoder')
        return outputs, state

    @staticmethod
    def _build_decoding_layer(mode,
                              cell_typ, num_layers, num_units, keep_prob,
                              embedded_targets, target_sequence_lengths,
                              target_embedding, batch_size, start_token_id, end_token_id,
                              encoder_state, output_layer, max_summary_length):
        """
        Create a training process in decoding layer
        :return: BasicDecoderOutput containing training logits and sample_id
        """

        cell = create_rnn_cell(cell_typ=cell_typ,
                               num_layers=num_layers,
                               num_units=num_units,
                               reuse=False,
                               keep_prob=keep_prob)

        if mode == 'train':
            helper = tf.contrib.seq2seq.TrainingHelper(input=embedded_targets,
                                                       sequence_length=target_sequence_lengths)
        elif mode == 'infer':
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=target_embedding,
                                                              start_tokens=tf.fill([batch_size], start_token_id),
                                                              end_token=end_token_id)
        else:
            raise ValueError

        decoder = tf.contrib.seq2seq.BasicDecoder(cell=cell,
                                                  helper=helper,
                                                  encoder_state=encoder_state,
                                                  output_layer=output_layer)

        # unrolling the decoder layer
        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,
                                                          impute_finished=True,
                                                          maximum_iterations=max_summary_length)
        return outputs

    def build_decoding_layer(self, target_vocab_index,
                             cell_typ, num_layers, num_units, keep_prob,
                             embedded_targets, target_sequence_lengths,
                             target_embedding, batch_size,
                             encoder_state, max_summary_length):
        vocab_size = len(target_vocab_index)
        start_token_id = target_vocab_index['SOS']
        end_token_id = target_vocab_index['EOS']

        output_layer = tf.layers.Dense(vocab_size)
        with tf.variable_scope('decoding'):
            decoding_output_train = self._build_decoding_layer(mode='train',
                                                               cell_typ=cell_typ, num_layers=num_layers,
                                                               num_units=num_units, keep_prob=keep_prob,
                                                               embedded_targets=embedded_targets,
                                                               target_sequence_lengths=target_sequence_lengths,
                                                               target_embedding=None, batch_size=None,
                                                               start_token_id=None, end_token_id=None,
                                                               encoder_state=encoder_state, output_layer=output_layer,
                                                               max_summary_length=max_summary_length)
        with tf.variable_scope('decoding', reuse=True):
            decoding_output_infer = self._build_decoding_layer(mode='infer',
                                                               cell_typ=cell_typ, num_layers=num_layers,
                                                               num_units=num_units, keep_prob=keep_prob,
                                                               embedded_targets=None,
                                                               target_sequence_lengths=None,
                                                               target_embedding=target_embedding, batch_size=batch_size,
                                                               start_token_id=start_token_id, end_token_id=end_token_id,
                                                               encoder_state=encoder_state, output_layer=output_layer,
                                                               max_summary_length=max_summary_length)

        return_dict = {'decoder_output_train': decoding_output_train,
                       'decoder_output_infer': decoding_output_infer
                       }
        return return_dict

    def _build_graph(self, placeholders):
        pass

    def _build_loss(self):
        pass

    def _build_optimizer(self):
        pass
