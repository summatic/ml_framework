import tensorflow as tf
import json


def parse_hparams():
    flags = tf.flags
    base_dir = '/home/hs/sandbox/query_expansion/custom_script_model/rbdm_4000white/'
    flags.DEFINE_string('base_dir', base_dir, 'path of base')

    # data configs
    flags.DEFINE_string('train_path', base_dir + 'train.txt', 'path of training file')
    flags.DEFINE_string('val_path', base_dir + 'val.txt', 'path of validation file')
    flags.DEFINE_string('test_path', base_dir + 'test.txt', 'path of testing file')
    flags.DEFINE_string('save_dir', base_dir + 'model', 'directory for saving logs/checkpoints')
    # flags.DEFINE_string('checkpoint_path', None, 'if True, load from checkpoint path')
    flags.DEFINE_string('checkpoint_path', base_dir + 'model/model.ckpt-40', 'if True, load from checkpoint path')
    flags.DEFINE_string('hparams_dir', base_dir + 'hparams.json', 'directory for saving hparams')

    # vocab configs
    # flags.DEFINE_string('pretrained_path', base_dir + 'pretrained.npy', 'path of vocabulary count file')
    flags.DEFINE_string('pretrained_path', None, 'path of vocabulary count file')
    flags.DEFINE_string('vocab_count_path', base_dir + 'word_count.txt', 'path of vocabulary count file')
    flags.DEFINE_string('vocab_index_path', base_dir + 'word_index.txt', 'path of vocabulary index file')
    flags.DEFINE_integer('min_count', 100, 'minimum count for pruning vocab')
    flags.DEFINE_integer('embedding_size', 256, 'embedding_size')

    # model(rnn) configs
    flags.DEFINE_integer('input_length', 20, 'input length')
    flags.DEFINE_string('cell_typ', 'LSTM', 'cell type')
    flags.DEFINE_integer('num_layers', 3, 'number of layers in stacked layer')
    flags.DEFINE_integer('num_units', 512, 'dimension of cell')
    flags.DEFINE_float('keep_prob', 0.2, 'keep prob')

    # model(cnn) configs
    flags.DEFINE_integer('pool_size', 3, 'pooling size')
    flags.DEFINE_integer('strides', 1, 'stride')
    flags.DEFINE_string('padding', 'valid', 'padding type')
    flags.DEFINE_string('data_format', 'channels_last', 'data format of pooling')

    # train configs
    flags.DEFINE_integer('batch_size', 256, 'batch size for training')
    flags.DEFINE_float('l2_loss', 0.1, 'coefficient of l2 loss')
    flags.DEFINE_integer('epochs', 10000, 'train epochs')
    flags.DEFINE_integer('steps_per_epochs', 100, 'steps per epochs')
    flags.DEFINE_integer('log_steps', 5, 'log_steps')
    flags.DEFINE_integer('summary_steps', 5, 'summary_steps')
    flags.DEFINE_integer('val_steps', 10, 'val_steps')

    # optimizer configs
    flags.DEFINE_string('optimizer', 'adam', 'optimizer')
    flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
    flags.DEFINE_float('beta1', 0.9, 'beta1')
    flags.DEFINE_float('beta2', 0.999, 'beta2')
    flags.DEFINE_float('epsilon', 1e-8, 'epsilon')
    flags.DEFINE_float('max_grad_norm', 1, 'max_grad_norm')

    # val & test configs
    flags.DEFINE_integer('val_size', 10000, 'val_size')

    return tf.contrib.training.HParams(**flags.FLAGS.flag_values_dict())


def save_hparams(hparams):
    with open(hparams.hparams_dir, 'w') as f:
        json.dump(hparams.to_json(), f)


def load_hparams(base_dir):
    hparams = parse_hparams()
    with open(base_dir + 'hparams.json', 'r', encoding='utf-8') as f:
        d = json.loads(json.loads(f.readline().strip()))

    hparams.override_from_dict(d)
    return hparams


def parse_test_hparams(base_dir):
    flags = tf.flags
    for name in list(flags.FLAGS):
        delattr(flags.FLAGS, name)
    flags.DEFINE_string('base_dir', base_dir, 'path of base')

    # data configs
    flags.DEFINE_string('train_path', base_dir + 'train.txt', 'path of training file')
    flags.DEFINE_string('val_path', base_dir + 'val.txt', 'path of validation file')
    flags.DEFINE_string('test_path', base_dir + 'test.txt', 'path of testing file')
    flags.DEFINE_string('save_dir', base_dir + 'model', 'directory for saving logs/checkpoints')
    flags.DEFINE_string('checkpoint_path', None, 'if True, load from checkpoint path')
    flags.DEFINE_string('hparams_dir', base_dir + 'hparams.json', 'directory for saving hparams')

    # vocab configs
    flags.DEFINE_string('pretrained_path', None, 'path of vocabulary count file')
    flags.DEFINE_string('vocab_count_path', base_dir + 'word_count.txt', 'path of vocabulary count file"')
    flags.DEFINE_string('vocab_index_path', base_dir + 'word_index.txt', 'path of vocabulary index file')
    flags.DEFINE_integer('min_count', 2, 'minimum count for pruning vocab')
    flags.DEFINE_integer('embedding_size', 8, 'embedding_size')

    # model(rnn) configs
    flags.DEFINE_integer('input_length', 4, 'input length')
    flags.DEFINE_string('cell_typ', 'LSTM', 'cell type')
    flags.DEFINE_integer('num_layers', 1, 'number of layers in stacked layer')
    flags.DEFINE_integer('num_units', 8, 'dimension of cell')
    flags.DEFINE_float('keep_prob', 0.3, 'keep prob')

    # model(cnn) configs
    flags.DEFINE_integer('pool_size', 3, 'pooling size')
    flags.DEFINE_integer('strides', 1, 'stride')
    flags.DEFINE_string('padding', 'valid', 'padding type')
    flags.DEFINE_string('data_format', 'channels_last', 'data format of pooling')

    # train configs
    flags.DEFINE_integer('batch_size', 2, 'batch size for training')
    flags.DEFINE_float('l2_loss', 0.1, 'coefficient of l2 loss')
    flags.DEFINE_integer('epochs', 10, 'train epochs')
    flags.DEFINE_integer('steps_per_epochs', 1, 'steps per epochs')
    flags.DEFINE_integer('log_steps', 2, 'log_steps')
    flags.DEFINE_integer('summary_steps', 4, 'log_steps')
    flags.DEFINE_integer('val_steps', 4, 'log_steps')

    # optimizer configs
    flags.DEFINE_string('optimizer', 'adam', 'optimizer')
    flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
    flags.DEFINE_float('beta1', 0.9, 'beta1 of optimizer')
    flags.DEFINE_float('beta2', 0.999, 'beta2 of optimizer')
    flags.DEFINE_float('epsilon', 1e-8, 'epsilon of optimizer')
    flags.DEFINE_float('max_grad_norm', 1, 'max gradient norm')

    # val configs

    # test configs

    hparams = tf.contrib.training.HParams(**flags.FLAGS.flag_values_dict())
    save_hparams(hparams=hparams)


if __name__ == '__main__':
    import os
    prefix = './tests/data/test_files/datasets/'
    base_dirs = [prefix + suffix for suffix in
                 ['labeled_pair/', 'labeled_single/', 'unlabeled_pair/', 'labeled_pair/']]

    for i in base_dirs:
        path = os.path.join(os.path.dirname(__file__), i)
        parse_test_hparams(path)
