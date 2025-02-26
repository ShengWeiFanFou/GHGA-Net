import json
from collections import defaultdict
import time

import joblib
from gensim.models import KeyedVectors
from paddlenlp.datasets import load_dataset
from tqdm import tqdm

from data_pre.doc_split import line_text_gen, entire_text_gen, entire_text_gen_sum, line_text_gen_sum, \
    line_text_gen_txt, entire_text_gen_txt
from data_pre.utils import tf_idf_out_pool
from data_pre.entire_word_graph_gen import load_stopwords, PMI


def read_dataset_rc(dataset_name,predict_data=None):
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
            line = []
            for c in text:
                line.append(c)
            seg_list.append(line)
        print(seg_list[0])
        json.dump(seg_list, open('../data_processed/CG/raw_character_list_{}.json'.format(dataset_name), 'w'))
        return
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
            line = []
            for c in text:
                line.append(c)
            seg_list.append(line)
        print(seg_list[0])
        json.dump(seg_list, open('../data_processed/CG/raw_character_list_{}.json'.format(dataset_name), 'w'))
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
            line = []
            for c in text:
                line.append(c)
            seg_list.append(line)
        print(seg_list[0])
        json.dump(seg_list, open('../data_processed/CG/raw_character_list_{}.json'.format(dataset_name), 'w'))
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
            line = []
            for c in text:
                line.append(c)
            seg_list.append(line)
        print(seg_list[0])
        json.dump(seg_list, open('../data_processed/CG/raw_character_list_{}.json'.format(dataset_name), 'w'))
        return
    elif dataset_name=='ours':
        doc_list = entire_text_gen_txt()
        seg_list=[]
        for text in tqdm(doc_list):
            line = []
            for c in text:
                line.append(c)
            seg_list.append(line)
        json.dump(seg_list, open('data/raw_character_list_{}.json'.format(dataset_name), 'w'))
        return
    elif dataset_name=='summary':
        doc_list = entire_text_gen_sum()
        seg_list=[]
        for text in tqdm(doc_list):
            line = []
            for c in text:
                line.append(c)
            seg_list.append(line)
        json.dump(seg_list, open('data/raw_character_list_{}.json'.format(dataset_name), 'w'))
        return
    elif dataset_name == 'predict':
        seg_list = []
        doc_list = predict_data
        for text in tqdm(doc_list):
            line = []
            for c in text:
                line.append(c)
            seg_list.append(line)
        json.dump(seg_list, open('data/raw_character_list_{}.json'.format(dataset_name), 'w'))
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
        line = []
        for c in text:
            line.append(c)
        seg_list.append(line)
    print(seg_list[0])
    json.dump(seg_list, open('../data_processed/CG/raw_character_list_{}.json'.format(dataset_name), 'w'))


def load_thucnews_short_rc():
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
    with open('../data/thucnews_short/train.txt', 'r', encoding='utf-8') as thu1:
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
        doc_list.append(item)
    for item in tqdm(test_text):
        doc_list.append(item)
    for item in tqdm(dev_text):
        doc_list.append(item)
    seg_list = []
    for text in tqdm(doc_list):
        line = []
        for c in text:
            line.append(c)
        seg_list.append(line)
    print(seg_list[0])
    json.dump(seg_list, open('../data_processed/CG/raw_character_list_ts.json', 'w'))


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
    stopwords = load_stopwords()
    label_dict = {}
    doc_list = []
    with open('../data/sample_data/train_file.txt', 'r', encoding='utf-8') as thu2:
        lines = thu2.readlines()
        # print(lines)
        for index, line in enumerate(lines):
            line = line.strip('\n')
            doc_list.append(line.split('\t')[1])
    with open('../data/sample_data/test_file.txt', 'r', encoding='utf-8') as thu1:
        lines = thu1.readlines()
        # print(lines)
        for index, line in enumerate(lines):
            line = line.strip('\n')
            doc_list.append(line.split('\t')[1])
            label = line.split('\t')[0]
            if label not in label_dict:
                label_dict[label] = len(label_dict)
    print(len(label_dict))
    seg_list = []
    for text in tqdm(doc_list):
        seg = [t for t in text if t not in stopwords and is_number(t) == 0]
        seg_list.append(seg)
    json.dump(seg_list, open('../data_processed/CG/raw_character_list_sample.json', 'w'))


def process_raw_data_c(num, dataset):
    word_freq = defaultdict(int)
    doc_list = json.load(open('data/raw_character_list_{}.json'.format(dataset), 'r'))
    stop_words = load_stopwords()
    for index, words in tqdm(enumerate(doc_list)):
        for word in words:
            if word not in stop_words and is_number(word)==0:
                word_freq[word] += 1
    print('总计词数：', len(word_freq))
    word_freq_10 = {}
    for key, value in word_freq.items():
        if 50 > value > num:
            word_freq_10[key] = value
    print('总计词数：', len(word_freq_10))
    # print(word_freq_10)
    json.dump(word_freq_10, open('data/character_freq_{}.json'.format(dataset), 'w'))


# 腾讯词向量
def equip_embed_c(dataset):
    word_freq_clean = json.load(open('data/character_freq_{}.json'.format(dataset), 'r'))
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
            found += 1
        # else:
        #     word_embed.append(np.zeros(200, dtype=np.float64))
        #     if word not in word_mapping:
        #         word_mapping[word] = len(word_mapping)

    print("词向量完整率为：", found / len(raw_word_list) * 100, "%")
    print('字典长度', len(word_mapping))
    word_list = []
    doc_list = json.load(open('data/raw_character_list_{}.json'.format(dataset), 'r'))
    for words in tqdm(doc_list):
        word = [one for one in words if one in word_mapping]
        word_list.append(' '.join(word))
    json.dump(word_list, open('data/{}_character_list.json'.format(dataset), 'w'))
    json.dump(word_mapping, open('data/{}_character_mapping.json'.format(dataset), 'w'))
    joblib.dump(word_embed, open('data/{}_character_emb_map.joblib'.format(dataset), 'wb'))


def cacluate_PMI_c(dataset):
    t = time.time()
    word_list = json.load(open('data/{}_character_list.json'.format(dataset), 'r'))
    word_mapping = json.load(open('data/{}_character_mapping.json'.format(dataset), 'r'))
    print(len(word_mapping))
    adj_word = PMI(word_list, word_mapping, window_size=5, sparse=True)
    adj_word = adj_word.toarray()
    print(adj_word.shape)
    # 大于4g不能用pickle存
    joblib.dump(adj_word, open('data/{}_adj_character.joblib'.format(dataset), 'wb'))
    print(time.time() - t)


def gen_rawchar_tnews(eval=False):
    data = load_dataset('clue', 'tnews')
    label_dict = {}
    for item in data['labels']:
        label_dict[item['label_desc']] = item['label']
    train_list = []
    test_list = []
    dev_list = []
    train_label = []
    test_label = []
    dev_label = []
    stop_word = load_stopwords()
    if eval:
        for i in tqdm(range(1)):
            seg = [c for c in data['train'][i]['sentence'] if c not in stop_word]
            train_list.append(seg)
        for i in tqdm(range(1)):
            seg = [c for c in data['train'][i]['sentence'] if c not in stop_word]
            train_list.append(seg)
        for i in tqdm(range(len(data['dev']))):
            seg = [c for c in data['dev'][i]['sentence'] if c not in stop_word]
            train_list.append(seg)
        json.dump(train_list, open('../data_processed/CG/tnews_qc_dict_eval.json', 'w'))
        return
    for i in tqdm(range(50000)):
        seg = [c for c in data['train'][i]['sentence'] if c not in stop_word]
        train_list.append(seg)
    for i in tqdm(range(50000, len(data['train']))):
        seg = [c for c in data['train'][i]['sentence'] if c not in stop_word]
        test_list.append(seg)
    for i in tqdm(range(len(data['dev']))):
        seg = [c for c in data['dev'][i]['sentence'] if c not in stop_word]
        dev_list.append(seg)
    for i in range(5):
        text = []
        for j in range(i * 10000, i * 10000 + 10000):
            text.append(train_list[i])
        for j in range(i * 672, i * 672 + 672):
            text.append(test_list[i])
        for j in range(3000):
            text.append(dev_list[i])
        json.dump(text, open('../data_processed/CG/tnews_qc_dict_{}.json'.format(i), 'w'))


def gen_rawchar_tnewsfew():
    data = load_dataset('fewclue', 'tnews')
    train_list = []
    test_list = []
    dev_list = []
    train_label = []
    test_label = []
    dev_label = []
    stopwords = load_stopwords()
    for i in tqdm(range(len(data['train_few_all']))):
        cs=[c for c in data['train_few_all'][i]['sentence'] if c not in stopwords and is_number(c) == 0]
        train_list.append(cs)
    # for i in tqdm(range(len(data['train_1']))):
    #     cs=[c for c in data['train_1'][i]['sentence'] if c not in stopwords and is_number(c) == 0]
    #     train_list.append(cs)
    # for i in tqdm(range(len(data['train_2']))):
    #     cs=[c for c in data['train_2'][i]['sentence'] if c not in stopwords and is_number(c) == 0]
    #     train_list.append(cs)
    for i in tqdm(range(len(data['test_public']))):
        cs=[c for c in data['test_public'][i]['sentence'] if c not in stopwords and is_number(c) == 0]
        test_list.append(cs)
    for i in tqdm(range(len(data['dev_few_all']))):
        cs=[c for c in data['dev_few_all'][i]['sentence'] if c not in stopwords and is_number(c) == 0]
        dev_list.append(cs)
    text = []
    for i in range(len(train_list)):
        text.append(train_list[i])
 
    for i in range(len(test_list)):
        text.append(test_list[i])

    for i in range(len(dev_list)):
        text.append(dev_list[i])

    json.dump(text, open('../data_processed/CG/tnews_qc_dict_0.json', 'w'))

    
def generate_rawchar_ours(data1):
    char_map = json.load(open('data/ours_character_mapping.json', 'r'))
    stopwords = load_stopwords()
    dict={}
    dict['confidential'] = []
    dict['secret'] = []
    dict['topsecret'] = []
    label_dict = {}
    train_list = []
    train_label = []
    dev_list = []
    dev_label = []
    test_list = []
    test_label = []
    if data1=='summary':
        train_l, test_l, dev_l, label_dict = line_text_gen_sum()
        for i in tqdm(range(len(train_l))):
            label, sen = train_l[i].split('\t')
            cs = [c for c in sen if c not in stopwords and is_number(c) == 0]
            cs = set(cs)
            cs = list(cs)
            train_list.append(cs)
            train_label.append(label_dict[label])
        for i in tqdm(range(len(test_l))):
            label, sen = test_l[i].split('\t')
            cs = [c for c in sen if c not in stopwords and is_number(c) == 0]
            cs = set(cs)
            cs = list(cs)
            test_list.append(cs)
            test_label.append(label_dict[label])
        for i in tqdm(range(len(dev_l))):
            label, sen = dev_l[i].split('\t')
            cs = [c for c in sen if c not in stopwords and is_number(c) == 0 ]
            cs = set(cs)
            cs = list(cs)
            dev_list.append(cs)
            dev_label.append(label_dict[label])

        text = []
        for i in range(len(train_list)):
            text.append(train_list[i])
        for i in range(len(dev_list)):
            text.append(dev_list[i])
        for i in range(len(test_list)):
            text.append(test_list[i])
        json.dump(text, open('data/summary_qc_dict_0.json', 'w'))

        return

    train_l,test_l,dev_l,label_dict=line_text_gen_txt()
    for i in tqdm(range(len(train_l))):
        label, sen = train_l[i].split('\t')
        cs = [c for c in sen if c not in stopwords and is_number(c) == 0 and c in char_map]
        cs=set(cs)
        cs=list(cs)
        train_list.append(cs)
        train_label.append(label_dict[label])
    for i in tqdm(range(len(test_l))):
        label, sen = test_l[i].split('\t')
        cs = [c for c in sen if c not in stopwords and is_number(c) == 0 and c in char_map]
        cs = set(cs)
        cs = list(cs)
        test_list.append(cs)
        test_label.append(label_dict[label])
    for i in tqdm(range(len(dev_l))):
        label, sen = dev_l[i].split('\t')
        cs = [c for c in sen if c not in stopwords and is_number(c) == 0 and c in char_map]
        cs = set(cs)
        cs = list(cs)
        dev_list.append(cs)
        dev_label.append(label_dict[label])
    # for i in tqdm(range(len(data))):
    #     cs=[c for c in data[i]['sentence'] if c not in stopwords and is_number(c) == 0]
    #     if data[i]['type']=='train':
    #         train_list.append(cs)
    #         train_label.append(label_dict[data[i]['label']])
    #     elif data[i]['type']=='test':
    #         dev_list.append(cs)
    #         dev_label.append(label_dict[data[i]['label']])
    text = []
    for i in range(len(train_list)):
        text.append(train_list[i])
    for i in range(len(dev_list)):
        text.append(dev_list[i])
    for i in range(len(test_list)):
        text.append(test_list[i])
    json.dump(text, open('data/ours_qc_dict_0.json', 'w'))
    

def generate_rawchar_predict(predict_data):
    char_map=json.load(open('data/ours_character_mapping.json', 'r'))
    train_list = []
    test_list = []
    dev_list = []
    stopwords = load_stopwords()
    seg = predict_data[0]
    cs = [c for c in seg if c not in stopwords and is_number(c) == 0 and c in char_map]
    train_list.append(cs)
    for i in tqdm(range(len(predict_data))):
        seg = predict_data[i]
        cs = [c for c in seg if c not in stopwords and is_number(c) == 0 and c in char_map]
        test_list.append(cs)
    seg = predict_data[1]
    cs = [c for c in seg if c not in stopwords and is_number(c) == 0 and c in char_map]
    dev_list.append(cs)
    text = []
    for i in range(len(train_list)):
        text.append(train_list[i])
    for i in range(len(dev_list)):
        text.append(dev_list[i])
    for i in range(len(test_list)):
        text.append(test_list[i])
    json.dump(text, open('data/predict_qc_dict_0.json', 'w'))


def gen_rawchar_iflytekfew():
    data = load_dataset('fewclue', 'iflytek')
    train_list = []
    test_list = []
    dev_list = []
    train_label = []
    test_label = []
    dev_label = []
    stopwords = load_stopwords()
    for i in tqdm(range(len(data['train_few_all']))):
        cs=[c for c in data['train_few_all'][i]['sentence'] if c not in stopwords and is_number(c) == 0]
        train_list.append(cs)
    for i in tqdm(range(len(data['test_public']))):
        cs=[c for c in data['test_public'][i]['sentence'] if c not in stopwords and is_number(c) == 0]
        test_list.append(cs)
    for i in tqdm(range(len(data['dev_few_all']))):
        cs=[c for c in data['dev_few_all'][i]['sentence'] if c not in stopwords and is_number(c) == 0]
        dev_list.append(cs)
    text = []
    for i in range(len(train_list)):
        text.append(train_list[i])
 
    for i in range(len(test_list)):
        text.append(test_list[i])

    for i in range(len(dev_list)):
        text.append(dev_list[i])

    json.dump(text, open('../data_processed/CG/iflytek_qc_dict_0.json', 'w'))
    

def gen_rawchar_csldcp():
    data = load_dataset('fewclue', 'csldcp')
    train_list = []
    test_list = []
    dev_list = []
    train_label = []
    test_label = []
    dev_label = []
    stopwords = load_stopwords()
    for i in tqdm(range(len(data['train_few_all']))):
        cs=[c for c in data['train_few_all'][i]['content'] if c not in stopwords and is_number(c) == 0]
        train_list.append(cs)
    for i in tqdm(range(len(data['test_public']))):
        cs=[c for c in data['test_public'][i]['content'] if c not in stopwords and is_number(c) == 0]
        test_list.append(cs)
    for i in tqdm(range(len(data['dev_few_all']))):
        cs=[c for c in data['dev_few_all'][i]['content'] if c not in stopwords and is_number(c) == 0]
        dev_list.append(cs)
    text = []
    for i in range(len(train_list)):
        text.append(train_list[i])
 
    for i in range(len(test_list)):
        text.append(test_list[i])

    for i in range(len(dev_list)):
        text.append(dev_list[i])

    json.dump(text, open('../data_processed/CG/csldcp_qc_dict_0.json', 'w'))
    
    
def generate_rc_sample():
    stopwords = load_stopwords()
    label_dict = {}
    train_list = []
    train_label = []
    dev_list = []
    dev_label = []
    with open('../data/sample_data/train_file.txt', 'r', encoding='utf-8') as thu2:
        lines = thu2.readlines()
        # print(lines)
        for line in lines:
            line = line.strip('\n')
            train_list.append(line.split('\t')[1])
            train_label.append(line.split('\t')[0])
    with open('../data/sample_data/test_file.txt', 'r', encoding='utf-8') as thu1:
        lines = thu1.readlines()
        # print(lines)
        for line in lines:
            line = line.strip('\n')
            dev_list.append(line.split('\t')[1])
            label = line.split('\t')[0]
            dev_label.append(label)
            if label not in label_dict:
                label_dict[label] = len(label_dict)
    print(label_dict)
    for i in range(1):
        index = 0
        text = []
        for j in range(500):
            seg = [c for c in train_list[j] if c not in stopwords and c != ' ' and is_number(c) == 0]
            text.append(seg)
            index += 1
        for j in range(500,1000):
            seg = [c for c in train_list[j] if c not in stopwords and c != ' ' and is_number(c) == 0]
            text.append(seg)
            index += 1
        for j in range(1000, 2000):
            seg = [c for c in train_list[j] if c not in stopwords and c != ' ' and is_number(c) == 0]
            text.append(seg)
            index += 1
        json.dump(text, open('../data_processed/CG/sample_qc_dict_{}.json'.format(i), 'w'))


def gen_rc_thucnews_short(i):
    label_dict = {}
    with open('../data/thucnews_short/class.txt', 'r', encoding='utf-8') as cla:
        ld = cla.readlines()
    for item in ld:
        item = item.strip('\n')
        if item not in label_dict:
            label_dict[item] = len(label_dict)
    print(label_dict)
    stop_word = load_stopwords()
    train_text = []
    train_label = []
    test_text = []
    test_label = []
    dev_text = []
    dev_label = []
    doc_list = []
    idx = []
    with open('../data/thucnews_short/train.txt', 'r', encoding='utf-8') as thu1:
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

    for index, item in enumerate(train_text):
        if index < 10000 * i:
            continue
        if index > 10000 * (i + 1):
            break
        seg = [c for c in item if c not in stop_word]
        doc_list.append(seg)
    for index, item in enumerate(test_text):
        if index > 1000:
            break
        seg = [c for c in item if c not in stop_word]
        doc_list.append(seg)
    for index, item in enumerate(dev_text):
        if index > 3000:
            break
        seg = [c for c in item if c not in stop_word]
        doc_list.append(seg)
    json.dump(doc_list, open('../data_processed/CG/ts_qc_dict_{}.json'.format(i), 'w'))

def gen_rawchar_eprstmt():
    data = load_dataset('fewclue', 'eprstmt')
    train_list = []
    test_list = []
    dev_list = []
    train_label = []
    test_label = []
    dev_label = []
    stopwords = load_stopwords()
    for i in tqdm(range(len(data['train_few_all']))):
        cs=[c for c in data['train_few_all'][i]['sentence'] if c not in stopwords and is_number(c) == 0]
        train_list.append(cs)
    for i in tqdm(range(len(data['test_public']))):
        cs=[c for c in data['test_public'][i]['sentence'] if c not in stopwords and is_number(c) == 0]
        test_list.append(cs)
    for i in tqdm(range(len(data['dev_few_all']))):
        cs=[c for c in data['dev_few_all'][i]['sentence'] if c not in stopwords and is_number(c) == 0]
        dev_list.append(cs)
    text = []
    for i in range(len(train_list)):
        text.append(train_list[i])
 
    for i in range(len(test_list)):
        text.append(test_list[i])

    for i in range(len(dev_list)):
        text.append(dev_list[i])

    json.dump(text, open('../data_processed/CG/eprstmt_qc_dict_0.json', 'w'))
    
    
def process_adj_c(dataset, i, eval=False):
    train_label_map, valid_label_map, test_label_map = {}, {}, {}
    label_list = []
    if eval:
        query_dict = json.load(open('../data_processed/CG/{}_qc_dict.json'.format(dataset), 'r'))
        filepath = '../data_processed/CG/adj_query2character_{}.joblib'.format(dataset)
    else:
        query_dict = json.load(open('data/{}_qc_dict_{}.json'.format(dataset, i), 'r'))
        filepath = 'data/adj_query2character_{}_{}.joblib'.format(dataset, i)
    word_list = []
    for line in query_dict:
        word_list.append(' '.join(line))
    word_mapping = json.load(open('data/ours_character_mapping.json'.format(dataset), 'r'))
    print(len(word_mapping))
    adj_query2word = tf_idf_out_pool(word_list, word_mapping, sparse=False)
    joblib.dump(adj_query2word, open(filepath, 'wb'))


if __name__ == '__main__':
    data='ours'
    read_dataset_rc(data)
    # load_thucnews_short_rc()
    # load_sample()
    process_raw_data_c(10,data)
    equip_embed_c(data)
    cacluate_PMI_c(data)
    # gen_rawchar_tnewsfew()
    # gen_rawchar_iflytekfew()
    generate_rawchar_ours()
    # # gen_rc_thucnews_short(1)
    process_adj_c(data, i=0, eval=0)
    # word_mapping = json.load(open('../data_processed/CG/csldcp_character_mapping.json', 'r'))
    # print(word_mapping)
