import sys
sys.path.extend(["../../", "../", "./"])
import random
import itertools
import argparse
from data.Vocab import *
from data.Dataloader import *
from driver.Config import *
import time
from modules.biLSTM import *
from modules.Bert import batch_data_bert_variable
from sklearn import metrics
import pickle


class Optimizer:
    def __init__(self, parameter, config, lr):
        self.optim = torch.optim.Adam(parameter, lr=lr, betas=(config.beta_1, config.beta_2),
                                      eps=config.epsilon, weight_decay=config.L2_REG)
        decay, decay_step = config.decay, config.decay_steps
        l = lambda epoch: decay ** (epoch // decay_step)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=l)

    def step(self):
        self.optim.step()
        self.schedule()
        self.optim.zero_grad()

    def schedule(self):
        self.scheduler.step()

    def zero_grad(self):
        self.optim.zero_grad()

    @property
    def lr(self):
        return self.scheduler.get_lr()


def train(train_inst, dev_data, test_data, model, vocab, config):
    model_param = filter(lambda p: p.requires_grad,
                         itertools.chain(
                             model.parameters(),
                         )
                         )

    model_optimizer = Optimizer(model_param, config, config.learning_rate)
    # model_optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    global_step = 0
    best_score = 0
    batch_num = int(np.ceil(len(train_inst) / float(config.train_batch_size)))

    for iter in range(config.train_iters):
        start_time = time.time()
        print('Iteration: ' + str(iter))
        batch_iter = 0

        overall_total_instance, overall_correct_instance = 0, 0
        for one_batch in data_iter(train_inst, config.train_batch_size, True):

            # bert_indice_input, bert_segments_ids, bert_piece_ids, bert_mask = \
            #     batch_pretrain_variable_sent_level(one_batch, vocab, config, tokenizer)
            # sent2span_index = batch_sent2span_offset(one_batch, config)

            batch_features, batch_features_length, batch_gold_label = batch_data_bert_variable(one_batch, vocab, config)

            model.train()
            batch_predict_output = model(batch_features, batch_features_length)

            loss = model.compute_loss(batch_predict_output, batch_gold_label)
            loss = loss / config.update_every
            loss_value = loss.data.cpu().numpy()
            loss.backward()

            total_instance, correct_instance = model.compute_accuracy(batch_predict_output, batch_gold_label)
            overall_total_instance += total_instance
            overall_correct_instance += correct_instance
            during_time = float(time.time() - start_time)
            acc = overall_correct_instance / overall_total_instance

            print("Step:%d, Iter:%d, batch:%d, time:%.2f, acc:%.2f, loss:%.2f"
                  % (global_step, iter, batch_iter, during_time, acc, loss_value))
            batch_iter += 1

            if batch_iter % config.update_every == 0 or batch_iter == batch_num:
                nn.utils.clip_grad_norm_(model_param, max_norm=config.clip)
                model_optimizer.step()
                model_optimizer.zero_grad()

                global_step += 1

            if batch_iter % config.validate_every == 0 or batch_iter == batch_num:
                print("Dev:")
                dev_score = predict(dev_data, model, vocab, config, config.dev_file + '.' + str(global_step))

                print("Test:")
                test_score = predict(test_data, model, vocab, config, config.test_file + '.' + str(global_step), True)

                if dev_score > best_score:
                    print("Exceed best Full F-score: history = %.4f, current = %.4f" % (best_score, dev_score))
                    print("test_score:", test_score)
                    best_score = dev_score
                    if 0 <= config.save_after <= iter:
                        explain_classify_model = {
                            "explain_classify": model.state_dict()
                        }
                        torch.save(explain_classify_model, config.save_model_path + "." + str(global_step))
                        print('Saving model to ', config.save_model_path + "." + str(global_step))


def predict(data, model, vocab, config, outputFile, test=False):
    start = time.time()
    model.eval()

    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    for one_batch in data_iter(data, config.test_batch_size, False):
        # bert_indice_input, bert_segments_ids, bert_piece_ids, bert_mask = \
        #     batch_pretrain_variable_sent_level(one_batch, vocab, config, tokenizer)

        batch_features, batch_features_length, batch_gold_label = batch_data_bert_variable(one_batch, vocab, config)

        # with torch.autograd.profiler.profile() as prof:
        batch_predict_output = model(batch_features, batch_features_length)

        batch_predict_label = torch.max(batch_predict_output.data, 1)[1].cpu().numpy()
        labels_all = np.append(labels_all, batch_gold_label.cpu().numpy())
        predict_all = np.append(predict_all, batch_predict_label)

    assert len(labels_all) == len(predict_all)
    accuracy = metrics.accuracy_score(labels_all, predict_all)
    f1 = metrics.f1_score(labels_all, predict_all, average='macro')

    outf = open(outputFile, mode='w', encoding='utf8')
    for index, predict in enumerate(predict_all):
        if predict != vocab.label2id(data[index].label):
            outf.write(
                '{0} gold:{1} predict:{2}\n'.format(data[index].segment, data[index].label, vocab.id2label(predict)))
    outf.close()
    end = time.time()
    during_time = float(end - start)
    print("samples num: %d,running time = %.2f ,accuracy: %.4f, f1: %.4f" % (len(data), during_time, accuracy, f1))
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=['1', '2', '3'], digits=5)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        print("Precision, Recall and F1-Score...")
        print(report)
        print("Confusion Matrix...")
        print(confusion)
    return f1


if __name__ == '__main__':
    random.seed(666)
    np.random.seed(666)
    torch.cuda.manual_seed(666)
    torch.manual_seed(666)

    ### gpu
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    print("CuDNN: \n", torch.backends.cudnn.enabled)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='config')
    argparser.add_argument('--dataset', required=True, help='choose a dataset')
    argparser.add_argument('--thread', default=4, type=int, help='thread num')
    argparser.add_argument('--use-cuda', action='store_true', default=True)

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)

    # train_data = read_corpus(config.train_file)
    # dev_data = read_corpus(config.dev_file)
    # test_data = read_corpus(config.test_file)

    if args.dataset == 'hotel':
        train_data, dev_data, test_data = read_slice_hotel_corpus(config)
    else:
        train_data, dev_data, test_data = read_slice_phone_corpus(config.test_file)

    vocab = creatVocab(train_data + dev_data + test_data, config.min_occur_count)
    # embedding = vocab.create_pretrained_embs(config)  # load embeddings

    torch.set_num_threads(args.thread)

    config.use_cuda = False
    if gpu and args.use_cuda: config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)

    start_a = time.time()

    train_insts = inst(train_data)
    dev_insts = inst(dev_data)
    test_insts = inst(test_data)

    print("train num: ", len(train_insts))
    print("dev num: ", len(dev_insts))
    print("test num: ", len(test_insts))

    if config.use_cuda:
        torch.backends.cudnn.enabled = True
        # torch.backends.cudnn.benchmark = True
        config.device = torch.device('cuda')
    else:
        config.device = torch.device('cpu')

    biLSTM = BiLSTM(config).to(config.device)

    train(train_insts, dev_insts, test_insts, biLSTM, vocab, config)
