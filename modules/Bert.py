from transformers import BertModel
import torch
from data.Dataloader import batch_slice
from torch.nn.utils.rnn import pad_sequence


def batch_data_bert_variable(batch, vocab, config):
    batch_gold_label = []
    batch.sort(key=lambda explain: explain.opinion.segment_length(), reverse=True)
    for explain in batch:
        batch_gold_label.append(vocab.label2id(explain.label))
    batch_features, batch_features_length = bert_pretraining(batch, config)
    batch_gold_label = torch.LongTensor(batch_gold_label).to(config.device)
    return batch_features, batch_features_length, batch_gold_label


def bert_pretraining(dataset, config):
    model = BertModel.from_pretrained('./bert-base-chinese')
    model.eval()
    model.to(config.device)

    batch_features = []
    batch_features_length = []

    for batch in batch_slice(dataset, config.bert_batch_size):
        tokens_tensor = []

        for explain in batch:
            tokens_tensor.append(torch.tensor(explain.opinion.ids))

        tokens_tensor = pad_sequence(tokens_tensor).T
        attention_mask = torch.ne(tokens_tensor, torch.zeros_like(tokens_tensor))

        tokens_tensor = tokens_tensor.to(config.device)
        attention_mask = attention_mask.to(config.device)

        with torch.no_grad():
            outputs = model(tokens_tensor, attention_mask=attention_mask)
            encoded_layers = outputs[0]

        for index, explain in enumerate(batch):
            batch_features.append(encoded_layers[index][explain.opinion.offset:explain.opinion.offset + explain.opinion.segment_length()])
            batch_features_length.append(explain.opinion.segment_length())

    batch_features = pad_sequence(batch_features)
    batch_features = torch.transpose(batch_features, 0, 1)
    batch_features_length = torch.tensor(batch_features_length).to(config.device)
    return batch_features, batch_features_length
