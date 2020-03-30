import re
from data.Tokenizer import Tokenizer


class Opinion:
    bert_tokenizer = Tokenizer('./bert-base-chinese'+'/vocab.txt')

    def __init__(self, statement, segment):
        self.sentence = re.sub('<(a|e|/)(\w|-)*?>', "", statement)
        if len(self.sentence) > 500:
            print('发现句子过长！')
            print(len(self.sentence), self.sentence)
        self.tokens = ['[CLS]']+self.bert_tokenizer.tokenize(self.sentence)+['[SEP]']
        self.ids = self.bert_tokenizer.convert_tokens_to_ids(self.tokens)
        self.segment_tokens = self.bert_tokenizer.tokenize(segment)
        self.offset = self.match_segment(self.segment_tokens)
        assert self.segment_tokens == self.tokens[self.offset:self.offset+self.segment_length()]

    def match_segment(self, segment_tokens):
        for i in range(len(self.tokens) - len(segment_tokens) + 1):
            index = i  # index指向下一个待比较的字符
            for j in range(len(segment_tokens)):
                if self.tokens[index] == segment_tokens[j]:
                    index += 1
                else:
                    break
                if index - i == len(segment_tokens):
                    return i
        raise ValueError(''.join(segment_tokens) + 'not in' + ''.join(self.tokens))

    def segment_length(self):
        return len(self.segment_tokens)

