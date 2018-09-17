import tensorflow as tf
import numpy as np


def create_rnn_cell(cell_typ, num_layers, num_units, reuse=None, keep_prob=0.1):
    cells = []
    for _ in range(num_layers):
        if cell_typ == 'RNN':
            cell = tf.contrib.rnn.BasicRNNCell(num_units, reuse=reuse)
        elif cell_typ == 'LSTM':
            cell = tf.contrib.rnn.BasicRNNCell(num_units, reuse=reuse)
        elif cell_typ == 'GRU':
            cell = tf.contrib.rnn.BasicRNNCell(num_units, reuse=reuse)
        else:
            raise ValueError

        if 0 < keep_prob < 1:
            cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob)
        cells.append(cell)

    return tf.contrib.rnn.MultiRNNCell(cells)


def create_bidirectional_dynamic_rnn(inputs, sequence_lengths, cell_typ, num_layers, num_units,
                                     reuse=None, keep_prob=0.1, scope=None):
    if scope is None:
        raise ValueError('Define scope')

    with tf.variable_scope(scope):
        cell_fw = create_rnn_cell(cell_typ=cell_typ,
                                  num_layers=num_layers,
                                  num_units=num_units,
                                  reuse=reuse,
                                  keep_prob=keep_prob)
        cell_bw = create_rnn_cell(cell_typ=cell_typ,
                                  num_layers=num_layers,
                                  num_units=num_units,
                                  reuse=reuse,
                                  keep_prob=keep_prob)

        outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                         cell_bw=cell_bw,
                                                         inputs=inputs,
                                                         sequence_length=sequence_lengths,
                                                         dtype=tf.float32,
                                                         scope=scope)

    return outputs, state


def create_embedding(pretrained_path,
                     name, vocab_size, embedding_size, trainable=True):
    if pretrained_path:
        embedding_npy = np.load(pretrained_path)
        initializer = tf.constant_initializer(embedding_npy.astype(np.float32))
        vocab_size, embedding_size = embedding_npy.shape
    else:
        initializer = tf.contrib.layers.xavier_initializer()
    embeddings = tf.get_variable(name=name,
                                 shape=(vocab_size, embedding_size),
                                 dtype=tf.float32,
                                 initializer=initializer,
                                 trainable=trainable)
    return embeddings


def remove_zero_vectors(tensor):
    """

    :param tensor: shape (x, y)
    :return:
    """
    _tensor = tf.reduce_sum(tensor, axis=1)
    zero_vector = tf.zeros(shape=(1, 1), dtype=tf.float32)
    bool_mask = tf.not_equal(_tensor, zero_vector)
    omit_zeros = tf.boolean_mask(tensor, tf.squeeze(bool_mask))
    return omit_zeros


def create_max_pooling1d(inputs, pool_size, strides, padding='valid', data_format='channels_last',
                         name=None, scope=None):
    if scope is None:
        raise ValueError('Define scope')

    with tf.variable_scope(scope):
        pooled = tf.layers.max_pooling1d(inputs=inputs,
                                         pool_size=pool_size,
                                         strides=strides,
                                         padding=padding,
                                         data_format=data_format,
                                         name=name)
    return pooled


def cosine_similarity(tensor1, tensor2):
    """

    :param tensor1: shape=(batch_size, dims)
    :param tensor2: shape=(batch_size, dims)
    :return:
    """

    norm1 = tf.nn.l2_normalize(tensor1, axis=1)
    norm2 = tf.nn.l2_normalize(tensor2, axis=1)

    return tf.reduce_sum(tf.multiply(norm1, norm2), axis=1)
