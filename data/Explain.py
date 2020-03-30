from data.Opinion import Opinion


class Explain:
    def __init__(self, segment, label, statement):
        self.segment = segment
        self.label = label
        self.opinion = Opinion(statement, segment)
        self.bert_embedding = None
