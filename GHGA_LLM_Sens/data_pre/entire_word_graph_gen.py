import json
import math
import time
from collections import defaultdict
import jieba.analyse as jn
import joblib
from numpy import float16
import numpy as np
from gensim.models import KeyedVectors
from scipy.sparse import coo_matrix
from tqdm import tqdm
import jieba
from paddlenlp.datasets import load_dataset

from data_pre.doc_split import entire_text_gen, entire_text_gen_sum, entire_text_gen_txt


# 根据整个数据集内容构建文档图
# 加载哈工大停用词表
def load_stopwords(filepath='data/hit_stopwords.txt'):
    stopwords = set()
    test = {}
    with open(filepath, 'r', encoding='utf-8') as sw:
        st = sw.readlines()
        for item in st:
            stopwords.add(item.replace('\n', ''))
    print(len(stopwords))
    return stopwords


# 读取数据集并分词
def read_dataset(dataset_name):
    if dataset_name == 'tnews' or dataset_name == 'iflytek':
        dataset = load_dataset('clue', dataset_name)
        text = 'sentence'
    else:
        dataset = load_dataset(dataset_name)
        text = 'text'
    doc_list = []
    for line in dataset['train']:
        doc_list.append(line[text])
    for line in dataset['test']:
        doc_list.append(line[text])
    for line in dataset['dev']:
        doc_list.append(line[text])
    seg_list = []
    for text in tqdm(doc_list):
        seg_list.append(jieba.lcut(text, cut_all=False))
    json.dump(seg_list, open('../data_processed/EWG/seg_list_{}.json'.format(dataset_name), 'w'))


# 读取数据集并分词
def read_dataset_tfidf(dataset_name,predict_data=None):
    if dataset_name == 'tnews':
        dataset = load_dataset('fewclue', dataset_name)
        text = 'sentence'
        doc_list = []
        for line in dataset['train_few_all']:
            doc_list.append(line[text])
        for line in dataset['test_public']:
            doc_list.append(line[text])
        for line in dataset['dev_few_all']:
            doc_list.append(line[text])
        for line in dataset['unlabeled']:
            doc_list.append(line[text])
        seg_list = []
        for text in tqdm(doc_list):
            seg_list.append(jn.extract_tags(text, topK=50, allowPOS=('n', 'ns', 'nt', 'vn', 'nw', 'nz','nr')))
        json.dump(seg_list, open('../data_processed/EWG/tf_idf_list_{}.json'.format(dataset_name), 'w'))
        return
        # label_dict = {}
        # for item in dataset['labels']:
        #     label_dict[item['label_desc']] = item['label']
        # label_num={}
        # label_list=[]
        # for item in dataset['train']:
        #     if item['label_desc'] not in label_num:
        #         label_num[item['label_desc']] = 1
        #     else:
        #         label_num[item['label_desc']] += 1
        # doc_list = []
        # for line in dataset['train']:
        #     doc_list.append(line[text])
        #     if label_num[line['label_desc']]<1200:
        #         label_list.append(1)
        #     else:
        #         label_list.append(0)
        # for line in dataset['test']:
        #     doc_list.append(line[text])
        #     label_list.append(0)
        # for line in dataset['dev']:
        #     doc_list.append(line[text])
        #     label_list.append(1)
        # seg_list = []
        # for text in tqdm(doc_list):
        #     seg_list.append(jn.extract_tags(text, topK=50, allowPOS=('n', 'ns', 'nt', 'vn', 'nw','nr')))
        # json.dump(seg_list, open('../data_processed/EWG/tf_idf_list_{}.json'.format(dataset_name), 'w'))
        # json.dump(label_list, open('../data_processed/EWG/numlabel_list_{}.json'.format(dataset_name), 'w'))
        # return
    elif dataset_name == 'iflytek':
        dataset = load_dataset('fewclue', dataset_name)
        text = 'sentence'
        doc_list = []
        for line in dataset['train_few_all']:
            doc_list.append(line[text])
        for line in dataset['test_public']:
            doc_list.append(line[text])
        for line in dataset['dev_few_all']:
            doc_list.append(line[text])
        for line in dataset['unlabeled']:
            doc_list.append(line[text])
        seg_list = []
        for text in tqdm(doc_list):
            seg_list.append(jn.extract_tags(text, topK=50, allowPOS=('n', 'ns', 'nt', 'vn', 'nw', 'nz','nr')))
        json.dump(seg_list, open('../data_processed/EWG/tf_idf_list_{}.json'.format(dataset_name), 'w'))
        return
    elif dataset_name == 'eprstmt':
        dataset = load_dataset('fewclue', dataset_name)
        text = 'sentence'
        doc_list = []
        for line in dataset['train_few_all']:
            doc_list.append(line[text])
        for line in dataset['test_public']:
            doc_list.append(line[text])
        for line in dataset['dev_few_all']:
            doc_list.append(line[text])
        for line in dataset['unlabeled']:
            doc_list.append(line[text])
        seg_list = []
        for text in tqdm(doc_list):
            seg_list.append(jn.extract_tags(text, topK=50, allowPOS=('n', 'ns', 'nt', 'vn', 'nw', 'nz','nr')))
        json.dump(seg_list, open('../data_processed/EWG/tf_idf_list_{}.json'.format(dataset_name), 'w'))
        return
    elif dataset_name == 'csldcp':
        dataset = load_dataset('fewclue', dataset_name)
        text = 'content'
        doc_list = []
        for line in dataset['train_few_all']:
            doc_list.append(line[text])
        for line in dataset['test_public']:
            doc_list.append(line[text])
        for line in dataset['dev_few_all']:
            doc_list.append(line[text])
        for line in dataset['unlabeled']:
            doc_list.append(line[text])
        seg_list = []
        for text in tqdm(doc_list):
            seg_list.append(jn.extract_tags(text, topK=50, allowPOS=('n', 'ns', 'nt', 'vn', 'nw', 'nz')))
        json.dump(seg_list, open('../data_processed/EWG/tf_idf_list_{}.json'.format(dataset_name), 'w'))
        return
    elif dataset_name == 'ours':
        seg_list = []
        doc_list = entire_text_gen_txt()
        for text in tqdm(doc_list):
            seg_list.append(jn.extract_tags(text, topK=50, allowPOS=('n', 'ns', 'nt', 'vn', 'nw', 'nz', 'nr')))
        json.dump(seg_list, open('data/tf_idf_list_{}.json'.format(dataset_name), 'w'))
        return
    elif dataset_name == 'summary':
        seg_list = []
        doc_list = entire_text_gen_sum()
        for text in tqdm(doc_list):
            seg_list.append(jn.extract_tags(text, topK=50, allowPOS=('n', 'ns', 'nt', 'vn', 'nw', 'nz', 'nr')))
        json.dump(seg_list, open('data/tf_idf_list_{}.json'.format(dataset_name), 'w'))
        return
    elif dataset_name == 'predict':
        seg_list = []
        doc_list = predict_data
        for text in tqdm(doc_list):
            seg_list.append(jn.extract_tags(text, topK=50, allowPOS=('n', 'ns', 'nt', 'vn', 'nw', 'nz', 'nr')))
        json.dump(seg_list, open('data/tf_idf_list_{}.json'.format(dataset_name), 'w'))
        return
    else:
        dataset = load_dataset(dataset_name)
        text = 'text'
    doc_list = []
    for line in dataset['train']:
        doc_list.append(line[text])
    for line in dataset['test']:
        doc_list.append(line[text])
    for line in dataset['dev']:
        doc_list.append(line[text])
    seg_list = []
    for text in tqdm(doc_list):
        seg_list.append(jn.extract_tags(text, topK=50, allowPOS=('n', 'ns', 'nt', 'vn', 'nw', 'nz', 'nr')))
    json.dump(seg_list, open('../data_processed/EWG/tf_idf_list_{}.json'.format(dataset_name), 'w'))


def load_ours():
    seg_list=[]
    doc_list=entire_text_gen(r'C:\Users\byz\Desktop\开题\美国国家档案馆数据集\数据集\dataset')
    for text in tqdm(doc_list):
        seg_list.append(jn.extract_tags(text, topK=50, allowPOS=('n', 'ns', 'nt', 'vn', 'nw', 'nz', 'nr')))
    json.dump(seg_list, open('../data_processed/EWG/tf_idf_list_ours.json', 'w'))
    
    
def load_thucnews_short():
    label_dict = {}
    with open('../data/thucnews_short/class.txt', 'r', encoding='utf-8') as cla:
        ld = cla.readlines()
    for item in ld:
        item = item.strip('\n')
        if item not in label_dict:
            label_dict[item] = len(label_dict)
    print(label_dict)
    train_text = []
    train_label = []
    test_text = []
    test_label = []
    dev_text = []
    dev_label = []
    doc_list = []
    with open('../data/thucnews_short/train.txt ', 'r', encoding='utf-8') as thu1:
        lines = thu1.readlines()
        # print(lines)
        for line in lines:
            line = line.strip('\n')
            train_text.append(line.split('\t')[0])
            train_label.append(line.split('\t')[-1])
    with open('../data/thucnews_short/test.txt', 'r', encoding='utf-8') as thu2:
        lines = thu2.readlines()
        # print(lines)
        for line in lines:
            line = line.strip('\n')
            test_text.append(line.split('\t')[0])
            test_label.append(line.split('\t')[-1])
    with open('../data/thucnews_short/dev.txt', 'r', encoding='utf-8') as thu3:
        lines = thu3.readlines()
        # print(lines)
        for line in lines:
            line = line.strip('\n')
            dev_text.append(line.split('\t')[0])
            dev_label.append(line.split('\t')[-1])
    # print(text)
    for item in tqdm(train_text):
        doc_list.append(jieba.lcut(item, cut_all=False))
    for item in tqdm(test_text):
        doc_list.append(jieba.lcut(item, cut_all=False))
    for item in tqdm(dev_text):
        doc_list.append(jieba.lcut(item, cut_all=False))
    json.dump(doc_list, open('../data_processed/EWG/tf_idf_list_ts.json', 'w'))


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def load_sample():
    label_dict = {}
    doc_list = []
    with open('../data/sample_data/train_file.txt', 'r', encoding='utf-8') as thu2: 
        lines = thu2.readlines()
        # print(lines)
        for index, line in enumerate(lines):
            if index>5000:
                break
            line = line.strip('\n')
            doc_list.append(line.split('\t')[1])
    with open('../data/sample_data/test_file.txt', 'r', encoding='utf-8') as thu1:
        lines = thu1.readlines()
        # print(lines)
        for index, line in enumerate(lines):
            line = line.strip('\n')
            # doc_list.append(line.split('\t')[1])
            label = line.split('\t')[0]
            if label not in label_dict:
                label_dict[label] = len(label_dict)
    print(len(label_dict))
    seg_list = []
    for text in tqdm(doc_list):
        seg_list.append(jn.extract_tags(text, topK=50, allowPOS=('n', 'ns', 'nt', 'vn ', 'nw', 'nz', 'nr')))
    json.dump(seg_list, open('../data_processed/EWG/tf_idf_list_sample.json', 'w'))


# 获取词频信息 只保留词频大于num的词汇
def process_raw_data(num, dataset,max=9999):
    word_freq = defaultdict(int)
    # doc_list=json.load(open('../data_processed/EWG/seg_list_{}.json'.format(dataset),'r'))
    doc_list = json.load(open('data/tf_idf_list_{}.json'.format(dataset), 'r'))
    # label_list = json.load(open('../data_processed/EWG/numlabel_list_{}.json'.format(dataset), 'r'))
    stop_words = load_stopwords()
    for index, words in tqdm(enumerate(doc_list)):
        for word in words:
            if word not in stop_words and len(word) > 1 and is_number(word) == 0:
                word_freq[word] += 1
            # if word not in stop_words and len(word)>1:
            #     if label_list[index]==1:
            #         word_freq[word]+=5
            #     else:
            #         word_freq[word] += 1
    print('总计词数：', len(word_freq))
    word_freq_10 = {}
    for key, value in word_freq.items():
        if num < value < max:
            word_freq_10[key] = value
    print('总计词数：', len(word_freq_10))
    json.dump(word_freq_10, open('data/word_freq_{}.json'.format(dataset), 'w'))


# 腾讯词向量
def equip_embed(dataset):
    word_freq_clean = json.load(open('data/word_freq_{}.json'.format(dataset), 'r'))
    raw_word_list = []
    word_embed = []
    word_mapping = {}
    for key in word_freq_clean.keys():
        raw_word_list.append(key)
    tic = time.time()
    wv_from_text = KeyedVectors.load('data/tencent_200d_emb_zh.bin', mmap='r')
    print('加载腾讯词向量时间 {:.2f}s'.format(time.time() - tic))
    found = 0
    w = tqdm(raw_word_list)
    for word in w:
        # 如果在词典之中，直接返回词向量,只匹配腾讯词向量中已有的词
        if word in wv_from_text.index_to_key:
            word_embed.append(wv_from_text[word])
            # 获取词map
            if word not in word_mapping:
                word_mapping[word] = len(word_mapping)
        else:
            word_embed.append(np.zeros(200, dtype=np.float64))
            if word not in word_mapping:
                word_mapping[word] = len(word_mapping)

    # print("词向量完整率为：", found / len(raw_word_list) * 100, "%")
    print('字典长度', len(word_mapping))
    word_list = []
    doc_list = json.load(open('data/tf_idf_list_{}.json'.format(dataset), 'r'))
    for words in tqdm(doc_list):
        word = [one for one in words if one in word_mapping]
        word_list.append(' '.join(word))
    json.dump(word_list, open('data/{}_word_list.json'.format(dataset), 'w'))
    json.dump(word_mapping, open('data/{}_word_mapping.json'.format(dataset), 'w'))
    joblib.dump(word_embed, open('data/{}_word_emb_map.joblib'.format(dataset), 'wb'))


# 计算词共现  如果map长度过大内存会溢出，只能改float精度
def PMI(inputs, mapping, window_size, sparse):
    W_ij = np.zeros([len(mapping), len(mapping)], dtype=float)
    W_i = np.zeros([len(mapping)], dtype=float)
    W_count = 0
    inpu = tqdm(inputs)
    for one in inpu:
        word_list = one.split(' ')
        if len(word_list) - window_size < 0:
            window_num = 1
        else:
            window_num = len(word_list) - window_size + 1

        for i in range(window_num):
            W_count += 1
            context = list(set(word_list[i:i + window_size]))
            while '' in context:
                context.remove('')
            for j in range(len(context)):
                W_i[mapping[context[j]]] += 1
                for k in range(j + 1, len(context)):
                    W_ij[mapping[context[j]], mapping[context[k]]] += 1
                    W_ij[mapping[context[k]], mapping[context[j]]] += 1

    if not sparse:
        PMI_adj = np.zeros([len(mapping), len(mapping)], dtype=np.float)
        for i in range(len(mapping)):
            for j in range(len(mapping)):
                PMI_adj[i, j] = math.log(W_ij[i, j] * W_count / W_i[i] / W_i[j]) if W_ij[i, j] != 0 else 0
                if i == j:
                    PMI_adj[i, j] = 1
                if PMI_adj[i, j] <= 0:
                    PMI_adj[i, j] = 0

    else:
        rows = []
        columns = []
        data = []
        for i in range(len(mapping)):
            for j in range(i, len(mapping)):
                value = math.log(W_ij[i, j] * W_count / W_i[i] / W_i[j]) if W_ij[i, j] != 0 else 0
                if i == j: value = 1

                if value > 0:
                    rows.append(i)
                    columns.append(j)
                    data.append(value)
                    if i != j:
                        rows.append(j)
                        columns.append(i)
                        data.append(value)

        PMI_adj = coo_matrix((data, (rows, columns)), shape=(len(mapping), len(mapping)))
    return PMI_adj


def cacluate_PMI(dataset):
    t = time.time()
    word_list = json.load(open('data/{}_word_list.json'.format(dataset), 'r'))
    word_mapping = json.load(open('data/{}_word_mapping.json'.format(dataset), 'r'))
    print(len(word_mapping))
    adj_word = PMI(word_list, word_mapping, window_size=5, sparse=True)
    adj_word = adj_word.toarray()
    print(adj_word.shape)
    # 大于4g不能用pickle存
    joblib.dump(adj_word, open('data/{}_adj_word.joblib'.format(dataset), 'wb'))
    print(time.time() - t)




if __name__ == '__main__':
    # read_dataset('thucnews')
    data = 'ours'
    # read_dataset_tfidf(data)
    # load_thucnews_short()
    # load_sample()
    # process_raw_data(0,data)
    # equip_embed(data)
    cacluate_PMI(data)
    # word_mapping = json.load(open('../data_processed/EWG/csldcp_word_mapping.json', 'r'))
    # print(word_mapping)