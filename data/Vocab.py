from collections import Counter
from gensim.models import KeyedVectors
import numpy as np


class Vocab(object):
    PAD, UNK = 0, 1

    def __init__(self, word_counter, label, min_occur_count=0):
        # self._id2word = ['<pad>', '<unk>']
        # self._wordid2freq = [10000, 10000]
        self._id2label = [k for k, v in label.most_common()]
        # for word, count in word_counter.most_common():
        #     if count > min_occur_count:
        #         self._id2word.append(word)
        #         self._wordid2freq.append(count)

        reverse = lambda x: dict(zip(x, range(len(x))))
        # self._word2id = reverse(self._id2word)
        self._label2id = reverse(self._id2label)
        # if len(self._word2id) != len(self._id2word):
        #     print("serious bug: words dumplicated, please check!")
        #
        # print("Vocab info: #words {0}".format(self.vocab_size))

    def create_pretrained_embs(self, config):
        word_vectors = KeyedVectors.load_word2vec_format(config.pretrained_embeddings_file, binary=False)
        wv_matrix = list()

        # one for UNK and one for zero padding
        wv_matrix.append(np.zeros(config.word_dims).astype("float32"))  # zero padding
        wv_matrix.append(np.random.uniform(-0.01, 0.01, config.word_dims).astype("float32"))  # UNK

        for word in self._id2word[2:]:
            if word in word_vectors.vocab:
                wv_matrix.append(word_vectors.word_vec(word))
            else:
                wv_matrix.append(wv_matrix[1])
        wv_matrix = np.array(wv_matrix)

        assert len(wv_matrix) == len(self._id2word)
        print('embedding size', wv_matrix.shape)
        return wv_matrix

    def word2id(self, xs):
        if isinstance(xs, list):
            return [self._word2id.get(x, self.UNK) for x in xs]
        return self._word2id.get(xs, self.UNK)

    def id2word(self, xs):
        if isinstance(xs, list):
            return [self._id2word[x] for x in xs]
        return self._id2word[xs]

    def id2label(self, xs):
        if isinstance(xs, list):
            return [self._id2label[x] for x in xs]
        return self._id2label[xs]

    def label2id(self, xs):
        if isinstance(xs, list):
            return [self._label2id.get(x, 0) for x in xs]
        return self._label2id.get(xs, 0)

    @property
    def vocab_size(self):
        return len(self._id2word)


def creatVocab(train_data, min_occur_count):
    word_counter = Counter()
    label = Counter()
    for instance in train_data:
        # for word in instance.seg_list:
        #     word_counter[word] += 1
        label[instance.label] += 1

    return Vocab(word_counter, label, min_occur_count)
