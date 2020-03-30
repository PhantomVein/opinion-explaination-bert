import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTM(nn.Module):
    def __init__(self, config):
        super(BiLSTM, self).__init__()

        self.lstm = nn.LSTM(config.word_dims, config.hidden_size, config.lstm_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout_lstm_hidden)
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)

    def forward(self, sequences, sequence_length):
        packed_embedded = nn.utils.rnn.pack_padded_sequence(sequences, sequence_length, batch_first=True)
        packed_outputs, hidden = self.lstm(packed_embedded)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True, padding_value=0.0)
        out = self.fc(torch.sum(outputs, 1))  # 句子开始时刻的 hidden state

        return out

    def compute_accuracy(self, predict_output, gold_label):
        predict_label = torch.max(predict_output.data, 1)[1].cpu()
        batch_size = len(predict_label)
        assert batch_size == len(gold_label)
        correct = sum(p == g for p, g in zip(predict_label, gold_label))
        return batch_size, correct.item()

    def compute_loss(self, predict_output, gold_label):
        loss = F.cross_entropy(predict_output, gold_label)
        return loss
