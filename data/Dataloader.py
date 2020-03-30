from data.Explain import Explain
import re
import collections
import numpy as np
import torch


def read_corpus(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        context = f.read()
        count = 1
        name = []
        item = []
        score = []
        comment = []
        polar = []
        for sentence in context.split('\n'):
            if count == 1:
                name.append(sentence)
            elif count == 2:
                item.append(sentence)
            elif count == 3:
                score.append(sentence)
            elif count == 4:
                comment.append(sentence)
            elif count == 5:
                polar.append(sentence)
            count = (count + 1) % 6
    for explain in explain_in_opinion(comment):
        data.append(explain)
    return data


def read_slice_hotel_corpus(config):
    train_data = read_corpus(config.train_file)
    dev_data = read_corpus(config.dev_file)
    test_data = read_corpus(config.test_file)

    print('\nloading data successfully')
    print('dataset:hotel')
    count = collections.Counter()
    for explain in train_data:
        count[explain.label] += 1
    print('train_data:\nfac:{0}\ncon:{1}\nsug:{2}\n'.format(count['fac'], count['con'], count['sug']))
    count = collections.Counter()
    for explain in dev_data:
        count[explain.label] += 1
    print('dev_data:\nfac:{0}\ncon:{1}\nsug:{2}\n'.format(count['fac'], count['con'], count['sug']))
    count = collections.Counter()
    for explain in test_data:
        count[explain.label] += 1
    print('test_data:\nfac:{0}\ncon:{1}\nsug:{2}\n'.format(count['fac'], count['con'], count['sug']))

    return train_data, dev_data, test_data


def read_slice_phone_corpus(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        context = f.read()
        product = []
        score = []
        comment = []
        polar = []
        for one_user in context.split('\n\n'):
            information_list = one_user.split('\n')
            if len(information_list) > 3:
                product.append(information_list[0])
                score.append(information_list[1])
                comment.extend(information_list[2:-1])
                polar.append(information_list[-1])
    for explain in explain_in_opinion(comment):
        data.append(explain)
    train_data = data[:len(data) // 10 * 7]
    dev_data = data[len(data) // 10 * 7:len(data) // 10 * 9]
    test_data = data[len(data) // 10 * 9:]

    print('dataset:phone')
    count = collections.Counter()
    for explain in train_data:
        count[explain.label] += 1
    print('train_data:\nfac:{0}\ncon:{1}\nsug:{2}\n'.format(count['fac'], count['con'], count['sug']))
    count = collections.Counter()
    for explain in dev_data:
        count[explain.label] += 1
    print('dev_data:\nfac:{0}\ncon:{1}\nsug:{2}\n'.format(count['fac'], count['con'], count['sug']))
    count = collections.Counter()
    for explain in test_data:
        count[explain.label] += 1
    print('test_data:\nfac:{0}\ncon:{1}\nsug:{2}\n'.format(count['fac'], count['con'], count['sug']))

    return train_data, dev_data, test_data


def explain_in_opinion(comment):
    for statement in comment:
        pattern_fac = re.compile(r'<exp-fac.*?>(.*?)</exp-fac.*?>')
        pattern_rea = re.compile(r'<exp-rea.*?>(.*?)</exp-rea.*?>')
        pattern_con = re.compile(r'<exp-con.*?>(.*?)</exp-con.*?>')
        pattern_sug = re.compile(r'<exp-sug.*?>(.*?)</exp-sug.*?>')
        factor = pattern_fac.findall(statement)
        reality = pattern_rea.findall(statement)
        condition = pattern_con.findall(statement)
        suggestion = pattern_sug.findall(statement)
        if len(factor + reality) != 0:
            for i in factor:
                yield Explain(i, 'fac', statement)
        if condition:
            for i in condition:
                yield Explain(i, 'con', statement)
        if suggestion:
            for i in suggestion:
                yield Explain(i, 'sug', statement)


def batch_slice(data, batch_size):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        sentences = [data[i * batch_size + b] for b in range(cur_batch_size)]

        yield sentences


def inst(data):
    return data


def data_iter(data, batch_size, shuffle=True):
    """
    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of  sentences in each batch
    """

    batched_data = []
    if shuffle: np.random.shuffle(data)
    batched_data.extend(list(batch_slice(data, batch_size)))

    if shuffle: np.random.shuffle(batched_data)
    for batch in batched_data:
        yield batch


def batch_data_variable(batch, vocab, config):
    batch_features = []
    batch_gold_label = []
    batch_features_length = []
    batch.sort(key=lambda explain: explain.length, reverse=True)
    for explain in batch:
        if explain.length < config.max_sentence_len:
            if not explain.regularized_seg:
                explain.regularized_seg = (
                        explain.seg_list + [vocab._id2word[0]] * (config.max_sentence_len - explain.length))
            batch_features_length.append(explain.length)
        else:
            if not explain.regularized_seg:
                explain.regularized_seg = explain.seg_list[:config.max_sentence_len]
            batch_features_length.append(config.max_sentence_len)
        # word to id
        if not explain.tokens:
            explain.tokens = vocab.word2id(explain.regularized_seg)
        batch_features.append(explain.tokens)
        batch_gold_label.append(vocab.label2id(explain.label))
    batch_features = torch.LongTensor(batch_features).to(config.device)
    batch_features_length = torch.IntTensor(batch_features_length).to(config.device)
    batch_gold_label = torch.LongTensor(batch_gold_label).to(config.device)
    return batch_features, batch_features_length, batch_gold_label


def batch_pretrain_variable_sent_level(batch, vocab, config, tokenizer):
    batch_size = len(batch)
    max_bert_len = -1
    max_sent_num = max([len(data[0].sentences) for data in batch])
    max_sent_len = max([len(sent) for data in batch for sent in data[0].sentences])
    # if config.max_sent_len < max_sent_len:max_sent_len = config.max_sent_len
    batch_bert_indices = []
    batch_segments_ids = []
    batch_piece_ids = []
    for data in batch:
        sents = data[0].sentences
        doc_bert_indices = []
        doc_semgents_ids = []
        doc_piece_ids = []
        for sent in sents:
            sent = sent[:max_sent_len]
            bert_indice, segments_id, piece_id = tokenizer.bert_ids(' '.join(sent))
            doc_bert_indices.append(bert_indice)
            doc_semgents_ids.append(segments_id)
            doc_piece_ids.append(piece_id)
            assert len(piece_id) == len(sent)
            assert len(bert_indice) == len(segments_id)
            bert_len = len(bert_indice)
            if bert_len > max_bert_len: max_bert_len = bert_len
        batch_bert_indices.append(doc_bert_indices)
        batch_segments_ids.append(doc_semgents_ids)
        batch_piece_ids.append(doc_piece_ids)
    bert_indice_input = np.zeros((batch_size, max_sent_num, max_bert_len), dtype=int)
    bert_mask = np.zeros((batch_size, max_sent_num, max_bert_len), dtype=int)
    bert_segments_ids = np.zeros((batch_size, max_sent_num, max_bert_len), dtype=int)
    bert_piece_ids = np.zeros((batch_size, max_sent_num, max_sent_len, max_bert_len), dtype=float)

    for idx in range(batch_size):
        doc_bert_indices = batch_bert_indices[idx]
        doc_semgents_ids = batch_segments_ids[idx]
        doc_piece_ids = batch_piece_ids[idx]
        sent_num = len(doc_bert_indices)
        assert sent_num == len(doc_semgents_ids)
        for idy in range(sent_num):
            bert_indice = doc_bert_indices[idy]
            segments_id = doc_semgents_ids[idy]
            bert_len = len(bert_indice)
            piece_id = doc_piece_ids[idy]
            sent_len = len(piece_id)
            assert sent_len <= bert_len
            for idz in range(bert_len):
                bert_indice_input[idx, idy, idz] = bert_indice[idz]
                bert_segments_ids[idx, idy, idz] = segments_id[idz]
                bert_mask[idx, idy, idz] = 1
            for idz in range(sent_len):
                for sid, piece in enumerate(piece_id):
                    avg_score = 1.0 / (len(piece))
                    for tid in piece:
                        bert_piece_ids[idx, idy, sid, tid] = avg_score

    bert_indice_input = torch.from_numpy(bert_indice_input)
    bert_segments_ids = torch.from_numpy(bert_segments_ids)
    bert_piece_ids = torch.from_numpy(bert_piece_ids).type(torch.FloatTensor)
    bert_mask = torch.from_numpy(bert_mask)

    return bert_indice_input, bert_segments_ids, bert_piece_ids, bert_mask
