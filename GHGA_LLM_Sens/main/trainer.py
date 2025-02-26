import random
import joblib
import numpy as np
import pickle as pkl
import json
import paddle
import paddle.nn.functional as F
import paddle.optimizer as optim
import time
import statistics
from paddle import int64
from pynvml import *
from sklearn import metrics
from tqdm import tqdm
import os

from main.gpu_util import show_gpu
from main.model import ETGAT


def fetch_tensor(tensor_dict, tensor_type, device):
    return paddle.to_tensor(tensor_dict[tensor_type], dtype='float32', place=device)


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    paddle.seed(seed)


class Trainer(object):
    def __init__(self, params):
        self.dataset = params.dataset
        self.max_epoch = params.max_epoch
        self.hidden_size = params.hidden_size
        self.device = params.device
        self.lr = params.lr
        self.weight_decay = params.weight_decay
        self.params = params
        self.data_path = params.data_path
        self.batch_size = params.batch_size
        self.trained = params.trained
        self.index = params.index
        self.do_eval = params.do_eval
        self.predict=params.predict
        self.adj_dict, self.features_dict, self.train_idx, self.valid_idx, self.test_idx, self.labels, word_num, char_num = self.load_data(
            self.data_path, self.index)
        self.label_num = 4
        self.labels = paddle.to_tensor(self.labels, place=self.device)
        word_emb_size = self.hidden_size * 2 + self.features_dict['word_emb'].shape[-1] + \
                        self.features_dict['char_emb'].shape[-1]
        # word_emb_size=456
        self.out_features_dim = [self.hidden_size, self.hidden_size, self.hidden_size, self.label_num]
        self.in_features_dim = [word_num, word_emb_size, self.hidden_size, self.hidden_size, char_num]
        print(self.out_features_dim)
        print(self.in_features_dim)
        self.train_idx = paddle.to_tensor(self.train_idx, place=self.device)
        self.model = ETGAT(self.in_features_dim, self.out_features_dim, self.train_idx, self.params)
        self.optim = optim.Adam(learning_rate=self.lr,
                                parameters=self.model.parameters(), weight_decay=self.weight_decay)
        if self.trained or self.predict:
            model_stat_dict = paddle.load('model_ours/model.pdparams'.format(self.dataset))
            opt_stat_dict = paddle.load('model_ours/opt.pdparams'.format(self.dataset))
            self.model.set_state_dict(model_stat_dict)
            self.optim.set_state_dict(opt_stat_dict)
            print('加载模型完成')

        self.model = self.model.to(self.device)
        total_trainable_params = sum(paddle.numel(p) for p in self.model.parameters())
        print(f'{total_trainable_params.item():,} training parameters.')

        paddle.fluid.core.get_int_stats()['STAT_gpu0_mem_size']  # 获取0号卡的显存分配信息
        show_gpu()

    def train(self):
        global_best_acc = 0
        global_best_f1 = 0
        global_best_epoch = 0
        best_test_acc = 0
        best_test_f1 = 0
        best_valid_epoch = 0
        best_valid_f1 = 0
        best_valid_acc = 0
        acc_valid = 0
        loss_valid = 0
        f1_valid = 0
        acc_test = 0
        loss_test = 0
        f1_test = 0
        for i in (range(1, self.max_epoch + 1)):
            t = time.time()
            output = self.model(self.adj_dict, self.features_dict)
            train_scores = output
            train_labels = self.labels[self.train_idx]
            loss_train = F.cross_entropy(train_scores, train_labels)
            self.optim.clear_grad()
            acc_train = paddle.cast(paddle.equal(paddle.argmax(train_scores, axis=-1), train_labels),
                                    dtype='float32').mean().item()
            loss_train.backward()
            self.optim.step()
            loss_train = loss_train.item()
            if i % 1 == 0:
                print('Epoch {}  loss: {:.4f} acc: {:.4f} time{:.4f}'.format(i, loss_train, acc_train, time.time() - t))
            text = []
            if i % 10 == 0:
                acc_valid, loss_valid, f1_valid, acc_test, loss_test, f1_test = self.test(i)
                if acc_test > global_best_acc:
                    global_best_acc = acc_test
                    global_best_f1 = f1_test
                    global_best_epoch = i
                    text.append('Epoch {}  loss: {:.4f} acc: {:.4f} time{:.4f}'.format(i, loss_train, acc_train,
                                                                                       time.time() - t))
                    text.append('test  f1: {:.4f} acc: {:.4f} loss: {:.4f}'.format(f1_test, acc_test, loss_test))
                    text.append('valid  f1: {:.4f} acc: {:.4f} loss: {:.4f}'.format(f1_valid, acc_valid, loss_valid))
                    self.save('./model_{}'.format(self.dataset), text)
                if acc_valid > best_valid_acc:
                    best_valid_acc = acc_valid
                    best_valid_f1 = f1_valid
                    best_test_acc = acc_test
                    best_test_f1 = f1_test
                    best_valid_epoch = i

                print('test  f1: {:.4f} acc: {:.4f} loss: {:.4f}'.format(f1_test, acc_test, loss_test))
                print('valid  f1: {:.4f} acc: {:.4f} loss: {:.4f}'.format(f1_valid, acc_valid, loss_valid))
                show_gpu()
                print('VALID: VALID ACC', best_valid_acc, ' VALID F1', best_valid_f1, 'EPOCH', best_valid_epoch)
                print('VALID: TEST ACC', best_test_acc, 'TEST F1', best_test_f1, 'EPOCH', best_valid_epoch)
                print('GLOBAL: TEST ACC', global_best_acc, 'TEST F1', global_best_f1, 'EPOCH', global_best_epoch)

        return best_test_acc, best_test_f1

    def test(self, epoch):
        t = time.time()
        self.model.training = False
        output = self.model.predict(self.adj_dict, self.features_dict)
        with paddle.no_grad():
            # valid_dict, valid_idx = self.generate_valid_batch(epoch, self.batch_size)
            valid_scores = output[self.valid_idx]
            valid_labels = self.labels[self.valid_idx]
            loss_valid = F.cross_entropy(valid_scores, valid_labels).item()
            acc_valid = paddle.cast(paddle.equal(paddle.argmax(valid_scores, axis=-1), valid_labels),
                                    dtype='float32').mean().item()
            f1_valid = metrics.f1_score(valid_labels.detach().cpu().numpy(),
                                        paddle.argmax(valid_scores, -1).detach().cpu().numpy(), average='macro')
            # test_dict, test_idx = self.generate_valid_batch(epoch, self.batch_size)
            test_scores = output[self.test_idx]
            test_labels = self.labels[self.test_idx]
            loss_test = F.cross_entropy(test_scores, test_labels).item()
            acc_test = paddle.cast(paddle.equal(paddle.argmax(test_scores, axis=-1), test_labels),
                                   dtype='float32').mean().item()
            f1_test = metrics.f1_score(test_labels.detach().cpu().numpy(),
                                       paddle.argmax(test_scores, -1).detach().cpu().numpy(), average='macro')
            # print('Valid  loss: {:.4f}  acc: {:.4f}  f1: {:.4f}'.format(loss_valid, acc_valid, f1_valid),
            # 'Test  loss: {:.4f} acc: {:.4f} f1: {:.4f} time: {:.4f}'.format(loss_test, acc_test, f1_test, time.time() - t))
        self.model.training = True
        # print('test time: ', time.time() - t)
        return acc_valid, loss_valid, f1_valid, acc_test, loss_test, f1_test

    # 模型预测过程
    def eval(self):
        predict_label=[]
        for i in range(1):
            self.model.training = False
            # 输出原始分数
            output = self.model.predict(self.adj_dict, self.features_dict)
            with paddle.no_grad():
                test_scores = output[self.test_idx]
                predict_score=test_scores.tolist()
                print(predict_score)
                for i in range(len(predict_score)):
                    al=[abs(x) for x in predict_score[i]]
                    predict_label.append(al.index(max(al)))
        # 取众数
        final_label=statistics.mode(predict_label)
        return final_label,predict_label

    def load_data(self, data_path, index):
        start = time.time()
        if 1:
            adj_query2word = joblib.load(
                open(data_path + 'adj_query2word_{}_{}.joblib'.format(self.dataset, index), 'rb'))
            adj_word = joblib.load(open(data_path + 'ours_adj_word.joblib'.format(self.dataset), 'rb'))
            adj_query2char = joblib.load(
                open(data_path + 'adj_query2character_{}_{}.joblib'.format(self.dataset, index), 'rb'))
            adj_char = joblib.load(open(data_path + 'ours_adj_character.joblib'.format(self.dataset), 'rb'))
            print(adj_char.shape[0])
            word_embs = joblib.load(open(data_path + 'ours_word_emb_map.joblib'.format(self.dataset), 'rb'))
            word_embs = np.array(word_embs, dtype=np.float32)
            char_embs = joblib.load(open(data_path + 'ours_character_emb_map.joblib'.format(self.dataset), 'rb'))
            char_embs = np.array(char_embs, dtype=np.float32)
            train_idx, valid_idx, test_idx = json.load(
                open(data_path + 'text_index_{}_{}.json'.format(self.dataset, index), 'r'))
            labels = json.load(open(data_path + 'labels_{}_{}.json'.format(self.dataset, index), 'r'))
            print('Length of [trian, valid, test, total]:',
                  [len(train_idx), len(valid_idx), len(test_idx), len(labels)])
            adj_dict, feature_dict = {}, {}
            adj_dict['q2w'] = adj_query2word
            adj_dict['word'] = adj_word
            word_num = adj_dict['q2w'].shape[1]
            feature_dict['word'] = np.eye(word_num, dtype=np.float32)
            feature_dict['word_emb'] = word_embs
            adj_dict['q2c'] = adj_query2char
            adj_dict['char'] = adj_char
            char_num = adj_dict['q2c'].shape[1]
            feature_dict['char'] = np.eye(char_num, dtype=np.float32)
            feature_dict['char_emb'] = char_embs
            adj, feature = {}, {}
            for i in adj_dict.keys():
                adj[i] = fetch_tensor(adj_dict, i, self.device)
            for i in feature_dict.keys():
                feature[i] = fetch_tensor(feature_dict, i, self.device)
            print('data process time: {}'.format(time.time() - start))
            return adj, feature, train_idx, valid_idx, test_idx, labels, word_num, char_num

        # 使用wiki外部语料库
        # adj_query2word = joblib.load(open(data_path + 'EG/adj_query2word_{}_{}.joblib'.format(self.dataset,index), 'rb'))
        # adj_word = joblib.load(open('../data/adj_word_wiki.joblib', 'rb'))
        # adj_query2char = joblib.load(open(data_path + 'CG/adj_query2character_{}_{}.joblib'.format(self.dataset,index), 'rb'))
        # adj_char = joblib.load(open(data_path + 'CG/{}_adj_character.joblib'.format(self.dataset), 'rb'))
        # print(adj_char.shape[0])
        # word_embs = pkl.load(open('../data/wiki_word_emb_map.pkl', 'rb'))
        # word_embs = np.array(word_embs, dtype=np.float32)
        # char_embs = joblib.load(open(data_path + 'CG/{}_character_emb_map.joblib'.format(self.dataset), 'rb'))
        # char_embs = np.array(char_embs, dtype=np.float32)
        # train_idx, valid_idx, test_idx = json.load(open(data_path + 'EG/text_index_{}_{}.json'.format(self.dataset,index), 'r'))
        # labels = json.load(open(data_path + 'EG/labels_{}_{}.json'.format(self.dataset,index), 'r'))
        # print('Length of [trian, valid, test, total]:', [len(train_idx), len(valid_idx), len(test_idx), len(labels)])
        # adj_dict, feature_dict = {}, {}
        # adj_dict['q2w'] = adj_query2word
        # adj_dict['word'] = adj_word
        # word_num = adj_dict['q2w'].shape[1]
        # feature_dict['word'] = np.eye(word_num, dtype=np.float32)
        # feature_dict['word_emb'] = word_embs
        # adj_dict['q2c'] = adj_query2char
        # adj_dict['char'] = adj_char
        # char_num = adj_dict['q2c'].shape[1]
        # feature_dict['char'] = np.eye(char_num, dtype=np.float32)
        # feature_dict['char_emb'] = char_embs
        # adj, feature = {}, {}
        # for i in adj_dict.keys():
        #     adj[i] = fetch_tensor(adj_dict, i, self.device)
        # for i in feature_dict.keys():
        #     feature[i] = fetch_tensor(feature_dict, i, self.device)
        # print('data process time: {}'.format(time.time() - start))
        # return adj, feature, train_idx, valid_idx, test_idx, labels, word_num,char_num

        adj_query2word = joblib.load(
            open(data_path + 'EG/adj_query2word_{}_{}.joblib'.format(self.dataset, index), 'rb'))
        adj_word = joblib.load(open(data_path + 'EWG/{}_adj_word.joblib'.format(self.dataset), 'rb'))
        adj_query2char = joblib.load(
            open(data_path + 'CG/adj_query2character_{}_{}.joblib'.format(self.dataset, index), 'rb'))
        adj_char = joblib.load(open(data_path + 'CG/{}_adj_character.joblib'.format(self.dataset), 'rb'))
        print(adj_char.shape[0])
        word_embs = joblib.load(open(data_path + 'EWG/{}_word_emb_map.joblib'.format(self.dataset), 'rb'))
        word_embs = np.array(word_embs, dtype=np.float32)
        char_embs = joblib.load(open(data_path + 'CG/{}_character_emb_map.joblib'.format(self.dataset), 'rb'))
        char_embs = np.array(char_embs, dtype=np.float32)
        train_idx, valid_idx, test_idx = json.load(
            open(data_path + 'EG/text_index_{}_{}.json'.format(self.dataset, index), 'r'))
        labels = json.load(open(data_path + 'EG/labels_{}_{}.json'.format(self.dataset, index), 'r'))
        print('Length of [trian, valid, test, total]:', [len(train_idx), len(valid_idx), len(test_idx), len(labels)])
        adj_dict, feature_dict = {}, {}
        adj_dict['q2w'] = adj_query2word
        adj_dict['word'] = adj_word
        word_num = adj_dict['q2w'].shape[1]
        feature_dict['word'] = np.eye(word_num, dtype=np.float32)
        feature_dict['word_emb'] = word_embs
        adj_dict['q2c'] = adj_query2char
        adj_dict['char'] = adj_char
        char_num = adj_dict['q2c'].shape[1]
        feature_dict['char'] = np.eye(char_num, dtype=np.float32)
        feature_dict['char_emb'] = char_embs
        adj, feature = {}, {}
        for i in adj_dict.keys():
            adj[i] = fetch_tensor(adj_dict, i, self.device)
        for i in feature_dict.keys():
            feature[i] = fetch_tensor(feature_dict, i, self.device)
        print('data process time: {}'.format(time.time() - start))
        return adj, feature, train_idx, valid_idx, test_idx, labels, word_num, char_num

    def save(self, path, text):
        # 保存模型的参数
        paddle.save(self.model.state_dict(), path + "/model.pdparams")
        # 保存优化器的参数
        paddle.save(self.optim.state_dict(), path + "/opt.pdparams")
        json.dump(text, open(path + "/best_log.json", 'w'))

    def load(self, path):
        model = paddle.load(path + './best_model_{}.pkl'.format(self.dataset))
        return model

    def generate_train_batch(self, batch_epoch, batch_size):
        index = batch_epoch * batch_size
        train_dit = self.adj_dict['q2w'][index:index + batch_size]
        train_idx = self.train_idx[index:index + batch_size]
        trian_dit = paddle.to_tensor(train_dit, place=self.device)
        trian_idx = paddle.to_tensor(train_idx, dtype=int64, place=self.device)
        return train_dit, train_idx

    def generate_test_batch(self, batch_epoch, batch_size):
        index = batch_epoch * batch_size
        test_dit = self.adj_dict['q2w'][index:index + batch_size]
        test_idx = self.train_idx[index:index + batch_size]
        test_dit = paddle.to_tensor(test_dit, place=self.device)
        test_idx = paddle.to_tensor(test_idx, dtype=int64, place=self.device)
        return test_dit, test_idx

    def generate_valid_batch(self, batch_epoch, batch_size):
        index = int(batch_epoch * batch_size / 2)
        batch_size = int(batch_size / 2)
        train_dit = self.adj_dict['q2w'][index:index + batch_size]
        train_idx = self.train_idx[index:index + batch_size]
        trian_dit = paddle.to_tensor(train_dit, place=self.device)
        trian_idx = paddle.to_tensor(train_idx, dtype=int64, place=self.device)
        return train_dit, train_idx


