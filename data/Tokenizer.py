import collections
import os


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip('\n')
        vocab[token] = index
    return vocab


def seg_char(sent):
    """
    把句子按字分开，不破坏英文结构
    """
    english = 'abcdefghijklmnopqrstuvwxyz0123456789'
    output = []
    buffer = ''
    for s in sent:
        if s in english or s in english.upper():
            buffer += s
        else:
            if buffer: output.append(buffer)
            buffer = ''
            output.append(s)
    if buffer: output.append(buffer)
    return output


class Tokenizer(object):

    def __init__(self, vocab_file):
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file))
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])

    def tokenize(self, sentence):
        chars = seg_char(sentence)
        return [x if x in self.vocab else self.ids_to_tokens[100] for x in chars]

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab[x] for x in tokens]
