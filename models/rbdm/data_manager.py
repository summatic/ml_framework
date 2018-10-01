from random import seed
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from data.data_manager import DataManager

seed(910417)


class RBDMDataManager(DataManager):

    def __init__(self, hparams):
        super().__init__(hparams=hparams)

    def _read_file(self, file_path, sep='\t'):
        f = open(file_path, 'r')

        id2raw = []
        id2indexed = []
        sent2id = {}
        data = []
        for n, line in enumerate(tqdm(f)):
            s0, s1 = line.strip().split(sep)

            if s0 not in sent2id:
                s0_id = len(sent2id)
                sent2id[s0] = s0_id
                id2raw.append(s0)
                id2indexed.append(self.indexing(s0))
            else:
                s0_id = sent2id[s0]

            if s1 not in sent2id:
                s1_id = len(sent2id)
                sent2id[s1] = s1_id
                id2raw.append(s1)
                id2indexed.append(self.indexing(s1))
            else:
                s1_id = sent2id[s1]

            s0_length = len(s0.split())
            s1_length = len(s1.split())

            data.append([s0_id, s1_id, s0_length, s1_length])

        data = np.array(data)
        id2indexed = np.array(id2indexed)

        f.close()

        return_dict = {'data': data, 'id2raw': id2raw, 'id2indexed': id2indexed, 'sent2id': sent2id}

        return return_dict

    def _build_placeholders(self):
        contexts = tf.placeholder(tf.int32, [self.hparams.batch_size, self.hparams.input_length],
                                  name='contexts')
        responses = tf.placeholder(tf.int32, [self.hparams.batch_size, self.hparams.input_length],
                                   name='responses')
        neg_responses = tf.placeholder(tf.int32, [self.hparams.batch_size, self.hparams.input_length],
                                       name='neg_responses')

        context_sequence_lengths = tf.placeholder(tf.int32, [self.hparams.batch_size],
                                                  name='context_sequence_lengths')
        response_sequence_lengths = tf.placeholder(tf.int32, [self.hparams.batch_size],
                                                   name='response_sequence_lengths')
        neg_response_sequence_lengths = tf.placeholder(tf.int32, [self.hparams.batch_size],
                                                       name='neg_response_sequence_lengths')

        return_dict = {
            'contexts': contexts,
            'responses': responses,
            'neg_responses': neg_responses,
            'context_sequence_lengths': context_sequence_lengths,
            'response_sequence_lengths': response_sequence_lengths,
            'neg_response_sequence_lengths': neg_response_sequence_lengths,
        }

        return return_dict

    def make_data_generator(self, mode):
        placeholders = self.placeholders

        dataset = self.dataset[mode]
        data = dataset['data']
        id2indexed = dataset['id2indexed']
        if mode == 'train':
            np.random.shuffle(data)
        else:
            sampled_id = np.random.choice(np.arange(len(data)), self.hparams.val_size)
            data = data[sampled_id]

        batch_size = self.hparams.batch_size
        for i in range(0, len(data), batch_size):
            _data = {k: [] for k, _ in placeholders.items()}
            for context_id, response_id, context_length, response_length in data[i:i+batch_size]:
                _data['contexts'].append(id2indexed[context_id])
                _data['responses'].append(id2indexed[response_id])
                _data['context_sequence_lengths'].append(context_length)
                _data['response_sequence_lengths'].append(response_length)

            if len(data[i:i+batch_size]) < batch_size:
                for _ in np.arange(batch_size - len(data[i:i+batch_size])):
                    _data['contexts'].append([0] * self.hparams.input_length)
                    _data['responses'].append([0] * self.hparams.input_length)
                    _data['context_sequence_lengths'].append(self.hparams.input_length)
                    _data['response_sequence_lengths'].append(self.hparams.input_length)

            if mode == 'train':
                neg_data_ids = np.random.choice(np.arange(len(data)), batch_size)
                neg_data = data[neg_data_ids]
                for _, neg_response_id, _, neg_response_length in neg_data:
                    _data['neg_responses'].append(id2indexed[neg_response_id])
                    _data['neg_response_sequence_lengths'].append(neg_response_length)
            else:
                for _ in range(batch_size):
                    _data['neg_responses'].append([0] * self.hparams.input_length)
                    _data['neg_response_sequence_lengths'].append(self.hparams.input_length)

            feed_dict = {placeholders[k]: v for k, v in _data.items()}

            yield feed_dict

    def make_instance(self, **kwargs):
        placeholders = self.placeholders
        contexts = kwargs['contexts']
        responses = kwargs['responses']

        data = []
        for context, response in zip(contexts, responses):
            context_sequence_length = len(context.split())
            response_sequence_length = len(response.split())

            data.append((self.indexing(context),
                         self.indexing(response),
                         context_sequence_length,
                         response_sequence_length))

        batch_size = self.hparams.batch_size
        for i in range(0, len(data), batch_size):
            _data = {k: [] for k, _ in placeholders.items()}
            for context, response, context_sequence_length, response_sequence_length in data[i:i+batch_size]:
                _data['contexts'].append(context)
                _data['responses'].append(response)
                _data['context_sequence_lengths'].append(context_sequence_length)
                _data['response_sequence_lengths'].append(response_sequence_length)
                _data['neg_responses'].append([0] * self.hparams.input_length)
                _data['neg_response_sequence_lengths'].append(self.hparams.input_length)

            if len(data[i:i+batch_size]) < batch_size:
                for _ in np.arange(batch_size - len(data[i:i+batch_size])):
                    _data['contexts'].append([0] * self.hparams.input_length)
                    _data['responses'].append([0] * self.hparams.input_length)
                    _data['context_sequence_lengths'].append(self.hparams.input_length)
                    _data['response_sequence_lengths'].append(self.hparams.input_length)
                    _data['neg_responses'].append([0] * self.hparams.input_length)
                    _data['neg_response_sequence_lengths'].append(self.hparams.input_length)

            feed_dict = {placeholders[k]: v for k, v in _data.items()}

            yield feed_dict
