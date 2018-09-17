from collections import defaultdict


predefined_index = {'UNK': 1, 'PAD': 0, 'SOS': 2, 'EOS': 3}


class VocabManager:

    def __init__(self, hparams):
        self.vocab_params = hparams

        self.vocab_counts = None
        self.vocab_index = None

    def build_vocab(self, sentences):
        """

        :param sentences: iterator of sentences
        :return:
        """
        vocab_counts = defaultdict(lambda: 0)
        for sentence in sentences:
            for vocab in sentence.split():
                vocab_counts[vocab] += 1
        vocab_counts = dict(vocab_counts)
        self.vocab_counts = vocab_counts

    def save_vocab_counts(self):
        """Save all built vocabs"""
        file_path = self.vocab_params.vocab_count_path

        f = open(file_path, 'w')
        for vocab, count in self.vocab_counts.items():
            f.write('{}\t{}\n'.format(vocab, count))
        f.close()

    def load_vocab_counts(self):
        """Load vocabs and its' count which count if bigger than min_count"""
        file_path = self.vocab_params.vocab_count_path
        min_count = self.vocab_params.min_count
        vocab_counts = {}

        f = open(file_path, 'r')
        for line in f:
            vocab, count = line.split('\t')
            count = int(count)

            if count < min_count:
                continue

            vocab_counts[vocab] = count
        self.vocab_counts = vocab_counts
        f.close()

    def index_vocabs(self):
        if self.vocab_counts is None:
            self.load_vocab_counts()

        vocab_index = predefined_index.copy()
        vocab_counts = sorted(self.vocab_counts.items(), key=lambda x: x[0])
        for vocab, _ in vocab_counts:
            vocab_index[vocab] = len(vocab_index)
        self.vocab_index = vocab_index

    def save_vocab_index(self):
        file_path = self.vocab_params.vocab_index_path

        f = open(file_path, 'w')
        for vocab, index in self.vocab_index.items():
            f.write('{}\t{}\n'.format(vocab, index))
        f.close()

    def load_vocab_index(self):
        """Load vocab index"""
        file_path = self.vocab_params.vocab_index_path
        vocab_index = {}

        f = open(file_path, 'r')
        for line in f:
            vocab, index = line.split('\t')
            vocab_index[vocab] = int(index)
        self.vocab_index = vocab_index
        f.close()
