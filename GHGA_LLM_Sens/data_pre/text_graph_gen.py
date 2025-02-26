import json
import pickle as pkl
import jieba.analyse as jn
import joblib
from paddlenlp.datasets import load_dataset
import jieba

from data_pre.doc_split import line_text_gen, line_text_gen_sum, line_text_gen_txt
from data_pre.utils import tf_idf_out_pool
from tqdm import tqdm
import os
import numpy as np
from data_pre.entire_word_graph_gen import load_stopwords


# 根据ernie模型预测的结果拿到实体列表
def equip_entity(dataset_name):
    list_index=json.load(open('../data_processed/EG/index_dict_{}.json'.format(dataset_name),'r'))
    print(list_index)
    res = json.load(open('../data_processed/EG/test_results_{}.json'.format(dataset_name),'r'))
    print(len(res['value']))
    list_entity=[]
    end = 0
    for index in list_index:
        entity=[]
        for i, item in enumerate(res['value']):
            if i<end:
                continue
            if i>index:
                list_entity.append(entity)
                end=i
                break
            for value in item:
                entity.append(value['entity'])
    # for line in list_entity:
    #     print(line)
    clean_entity = []
    for line in list_entity:
        ce = []
        for word in line:
            if type(word) == list or word == '':
                continue
            ce.append(word)
        clean_entity.append(ce)
    json.dump(clean_entity,open('../data_processed/EG/entity_list_{}.json'.format(dataset_name),'w'))


def gen_rawtext_tnews(eval=False):
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
            seg = jieba.lcut(data['train'][i]['sentence'], cut_all=False)
            line = []
            for s in seg:
                if s not in stop_word and len(s) > 1:
                    line.append(s)
            train_list.append(line)
            train_label.append(data['train'][i]['label'])
        for i in tqdm(range(1)):
            seg = jieba.lcut(data['train'][i]['sentence'], cut_all=False)
            line = []
            for s in seg:
                if s not in stop_word and len(s) > 1:
                    line.append(s)
            dev_list.append(line)
            dev_label.append(data['train'][i]['label'])
        for i in tqdm(range(len(data['dev']))):
            seg = jieba.lcut(data['dev'][i]['sentence'], cut_all=False)
            line = []
            for s in seg:
                if s not in stop_word and len(s) > 1:
                    line.append(s)
            test_list.append(line)
            test_label.append(data['dev'][i]['label'])
        text = []
        idx = []
        index = 0
        for i in range(len(train_list)):
            text.append(train_list[i])
            idx.append(str(index) + ' train ' + str(train_label[i]))
            index += 1
        for i in range(len(dev_list)):
            text.append(dev_list[i])
            idx.append(str(index) + ' dev ' + str(dev_label[i]))
            index += 1
        for i in range(len(test_list)):
            text.append(test_list[i])
            idx.append(str(index) + ' test ' + str(test_label[i]))
            index += 1
        json.dump(text, open('../data_processed/EG/tnews_query_dict_eval.json', 'w'))
        json.dump(idx, open('../data_processed/EG/split_label_tnews_eval.json', 'w'))
        return
    for i in tqdm(range(50000)):
        seg = jieba.lcut(data['train'][i]['sentence'], cut_all=False)
        line = []
        for s in seg:
            if s not in stop_word and len(s) > 1:
                line.append(s)
        train_list.append(line)
        train_label.append(data['train'][i]['label'])
    for i in tqdm(range(50000, len(data['train']))):
        seg = jieba.lcut(data['train'][i]['sentence'], cut_all=False)
        line = []
        for s in seg:
            if s not in stop_word and len(s) > 1:
                line.append(s)
        test_list.append(line)
        test_label.append(data['train'][i]['label'])
    for i in tqdm(range(len(data['dev']))):
        seg = jieba.lcut(data['dev'][i]['sentence'], cut_all=False)
        line = []
        for s in seg:
            if s not in stop_word and len(s) > 1:
                line.append(s)
        dev_list.append(line)
        dev_label.append(data['dev'][i]['label'])
    for i in range(5):
        text = []
        idx = []
        index = 0
        for j in range(i * 10000, i * 10000 + 10000):
            text.append(train_list[j])
            idx.append(str(index) + ' train ' + str(train_label[j]))
            index += 1
        for j in range(i * 672, i * 672 + 672):
            text.append(test_list[j])
            idx.append(str(index) + ' test ' + str(test_label[j]))
            index += 1
        for j in range(3000):
            text.append(dev_list[j])
            idx.append(str(index) + ' dev ' + str(dev_label[j]))
            index += 1
        json.dump(text, open('../data_processed/EG/tnews_query_dict_{}.json'.format(i), 'w'))
        json.dump(idx, open('../data_processed/EG/split_label_tnews_{}.json'.format(i), 'w'))


def gen_thucnews_short(i):
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
        seg = jieba.lcut(item, cut_all=False)
        line = []
        for s in seg:
            if s not in stop_word and len(s) > 1:
                line.append(s)
        doc_list.append(line)
        idx.append(str(index - 10000 * i) + ' train ' + str(train_label[index]))
    for index, item in enumerate(test_text):
        if index > 1000:
            break
        seg = jieba.lcut(item, cut_all=False)
        line = []
        for s in seg:
            if s not in stop_word and len(s) > 1:
                line.append(s)
        doc_list.append(line)
        idx.append(str(index + 10001) + ' test ' + str(test_label[index]))
    for index, item in enumerate(dev_text):
        if index > 3000:
            break
        seg = jieba.lcut(item, cut_all=False)
        line = []
        for s in seg:
            if s not in stop_word and len(s) > 1:
                line.append(s)
        doc_list.append(line)
        idx.append(str(index + 11002) + ' dev ' + str(dev_label[index]))
    json.dump(doc_list, open('../data_processed/EG/ts_query_dict_{}.json'.format(i), 'w'))
    json.dump(idx, open('../data_processed/EG/split_label_ts_{}.json'.format(i), 'w'))


def gen_rawtext_iflytek():
    data = load_dataset('clue', 'iflytek')
    label_dict = {}
    for item in data['labels']:
        label_dict[item['label_des']] = item['label']
    train_list = []
    test_list = []
    dev_list = []
    train_label = []
    test_label = []
    dev_label = []
    for i in tqdm(range(len(data['train']))):
        seg = jn.extract_tags(data['train'][i]['sentence'], topK=6, allowPOS=('n', 'ns', 'nt', 'vn', 'nw'))
        train_list.append(seg)
        train_label.append(data['train'][i]['label'])
    for i in tqdm(range(5)):
        seg = jn.extract_tags(data['train'][i]['sentence'], topK=6, allowPOS=('n', 'ns', 'nt', 'vn', 'nw'))
        test_list.append(seg)
        test_label.append(data['train'][i]['label'])
    for i in tqdm(range(len(data['dev']))):
        seg = jn.extract_tags(data['dev'][i]['sentence'], topK=6, allowPOS=('n', 'ns', 'nt', 'vn', 'nw'))
        dev_list.append(seg)
        dev_label.append(data['dev'][i]['label'])
    text = []
    idx = []
    index = 0
    for i in range(len(train_list)):
        text.append(train_list[i])
        idx.append(str(index) + ' train ' + str(train_label[i]))
        index += 1
    for i in range(len(test_list)):
        text.append(test_list[i])
        idx.append(str(index) + ' test ' + str(test_label[i]))
        index += 1
    for i in range(len(dev_list)):
        text.append(dev_list[i])
        idx.append(str(index) + ' dev ' + str(dev_label[i]))
        index += 1
    json.dump(text, open('../data_processed/EG/iflytek_query_dict_0.json', 'w'))
    json.dump(idx, open('../data_processed/EG/split_label_iflytek_0.json', 'w'))


def gen_rawtext_csldcp():
    data = load_dataset('fewclue', 'csldcp')
    label_dict = {}
    for item in data['train_few_all']:
        if item['label'] not in label_dict:
            label_dict[item['label']] = len(label_dict)
    train_list = []
    test_list = []
    dev_list = []
    train_label = []
    test_label = []
    dev_label = []
    for i in tqdm(range(len(data['train_few_all']))):
        seg = jn.extract_tags(data['train_few_all'][i]['content'], topK=6, allowPOS=('n', 'ns', 'nt', 'vn', 'nw'))
        train_list.append(seg)
        train_label.append(label_dict[data['train_few_all'][i]['label']])
    for i in tqdm(range(len(data['test_public']))):
        seg = jn.extract_tags(data['test_public'][i]['content'], topK=6, allowPOS=('n', 'ns', 'nt', 'vn', 'nw'))
        test_list.append(seg)
        test_label.append(label_dict[data['test_public'][i]['label']])
    for i in tqdm(range(len(data['dev_few_all']))):
        seg = jn.extract_tags(data['dev_few_all'][i]['content'], topK=6, allowPOS=('n', 'ns', 'nt', 'vn', 'nw'))
        dev_list.append(seg)
        dev_label.append(label_dict[data['dev_few_all'][i]['label']])
    text = []
    idx = []
    index = 0
    for i in range(len(train_list)):
        text.append(train_list[i])
        idx.append(str(index) + ' train ' + str(train_label[i]))
        index += 1
    for i in range(len(test_list)):
        text.append(test_list[i])
        idx.append(str(index) + ' test ' + str(test_label[i]))
        index += 1
    for i in range(len(dev_list)):
        text.append(dev_list[i])
        idx.append(str(index) + ' dev ' + str(dev_label[i]))
        index += 1
    json.dump(text, open('../data_processed/EG/csldcp_query_dict_0.json', 'w'))
    json.dump(idx, open('../data_processed/EG/split_label_csldcp_0.json', 'w'))


def gen_rawtext_tnewsfew():
    data = load_dataset('fewclue', 'tnews')
    label_dict = {}
    for item in data['train_few_all']:
        if item['label_desc'] not in label_dict:
            label_dict[item['label_desc']] = len(label_dict)
    train_list = []
    test_list = []
    dev_list = []
    train_label = []
    test_label = []
    dev_label = []
    stopwords = load_stopwords()
    for i in tqdm(range(len(data['train_few_all']))):
        seg = jieba.lcut(data['train_few_all'][i]['sentence'], cut_all=False)
        cs=[c for c in seg if c not in stopwords and len(c) > 1 and is_number(c) == 0]
        train_list.append(cs)
        train_label.append(label_dict[data['train_few_all'][i]['label_desc']])
    # for i in tqdm(range(len(data['train_1']))):
    #     seg = jieba.lcut(data['train_1'][i]['sentence'], cut_all=False)
    #     cs=[c for c in seg if c not in stopwords and len(c) > 1 and is_number(c) == 0]
    #     train_list.append(cs)
    #     train_label.append(label_dict[data['train_1'][i]['label_desc']])
    # for i in tqdm(range(len(data['train_2']))):
    #     seg = jieba.lcut(data['train_2'][i]['sentence'], cut_all=False)
    #     cs=[c for c in seg if c not in stopwords and len(c) > 1 and is_number(c) == 0]
    #     train_list.append(cs)
    #     train_label.append(label_dict[data['train_2'][i]['label_desc']])
    for i in tqdm(range(len(data['test_public']))):
        seg = jieba.lcut(data['test_public'][i]['sentence'], cut_all=False)
        cs=[c for c in seg if c not in stopwords and len(c) > 1 and is_number(c) == 0]
        test_list.append(cs)
        test_label.append(label_dict[data['test_public'][i]['label_desc']])
    for i in tqdm(range(len(data['dev_few_all']))):
        seg = jieba.lcut(data['dev_few_all'][i]['sentence'], cut_all=False)
        cs=[c for c in seg if c not in stopwords and len(c) > 1 and is_number(c) == 0]
        dev_list.append(cs)
        dev_label.append(label_dict[data['dev_few_all'][i]['label_desc']])
    text = []
    idx = []
    index = 0
    for i in range(len(train_list)):
        text.append(train_list[i])
        idx.append(str(index) + ' train ' + str(train_label[i]))
        index += 1
    for i in range(len(test_list)):
        text.append(test_list[i])
        idx.append(str(index) + ' test ' + str(test_label[i]))
        index += 1
    for i in range(len(dev_list)):
        text.append(dev_list[i])
        idx.append(str(index) + ' dev ' + str(dev_label[i]))
        index += 1
    json.dump(text, open('../data_processed/EG/tnews_query_dict_0.json', 'w'))
    json.dump(idx, open('../data_processed/EG/split_label_tnews_0.json', 'w'))
    
    
def generate_text_ours(data):
    train_list = []
    test_list = []
    dev_list = []
    train_label = []
    test_label = []
    dev_label = []
    stopwords = load_stopwords()

    if data=='summary':
        train_l, test_l, dev_l, label_dict = line_text_gen_sum()
        for i in tqdm(range(len(train_l))):
            label, sen = train_l[i].split('\t')
            seg = jn.extract_tags(sen, topK=10, allowPOS=('n', 'ns', 'nt', 'vn', 'nw', 'nz', 'nr'))
            cs = [c for c in seg if c not in stopwords and len(c) > 1 and is_number(c) == 0]
            train_list.append(cs)
            train_label.append(label_dict[label])
        for i in tqdm(range(len(test_l))):
            label, sen = test_l[i].split('\t')
            seg = jn.extract_tags(sen, topK=10, allowPOS=('n', 'ns', 'nt', 'vn', 'nw', 'nz', 'nr'))
            cs = [c for c in seg if c not in stopwords and len(c) > 1 and is_number(c) == 0]
            test_list.append(cs)
            test_label.append(label_dict[label])
        for i in tqdm(range(len(dev_l))):
            label, sen = dev_l[i].split('\t')
            seg = jn.extract_tags(sen, topK=10, allowPOS=('n', 'ns', 'nt', 'vn', 'nw', 'nz', 'nr'))
            cs = [c for c in seg if c not in stopwords and len(c) > 1 and is_number(c) == 0]
            dev_list.append(cs)
            dev_label.append(label_dict[label])
        text = []
        idx = []
        index = 0
        for i in range(len(train_list)):
            text.append(train_list[i])
            idx.append(str(index) + ' train ' + str(train_label[i]))
            index += 1
        for i in range(len(dev_list)):
            text.append(dev_list[i])
            idx.append(str(index) + ' dev ' + str(dev_label[i]))
            index += 1
        for i in range(len(test_list)):
            text.append(test_list[i])
            idx.append(str(index) + ' test ' + str(test_label[i]))
            index += 1
        json.dump(text, open('data/summary_query_dict_0.json', 'w'))
        json.dump(idx, open('data/split_label_summary_0.json', 'w'))
        return

    train_l,test_l,dev_l,label_dict=line_text_gen_txt()
    for i in tqdm(range(len(train_l))):
        label,sen=train_l[i].split('\t')
        seg = jn.extract_tags(sen, topK=10, allowPOS=('n', 'ns', 'nt', 'vn', 'nw', 'nz', 'nr'))
        cs=[c for c in seg if c not in stopwords and len(c) > 1 and is_number(c) == 0]
        train_list.append(cs)
        train_label.append(label_dict[label])
    for i in tqdm(range(len(test_l))):
        label, sen = test_l[i].split('\t')
        seg = jn.extract_tags(sen, topK=10, allowPOS=('n', 'ns', 'nt', 'vn', 'nw', 'nz', 'nr'))
        cs = [c for c in seg if c not in stopwords and len(c) > 1 and is_number(c) == 0]
        test_list.append(cs)
        test_label.append(label_dict[label])
    for i in tqdm(range(len(dev_l))):
        label, sen = dev_l[i].split('\t')
        seg = jn.extract_tags(sen, topK=10, allowPOS=('n', 'ns', 'nt', 'vn', 'nw', 'nz', 'nr'))
        cs = [c for c in seg if c not in stopwords and len(c) > 1 and is_number(c) == 0]
        dev_list.append(cs)
        dev_label.append(label_dict[label])
    text = []
    idx = []
    index = 0
    for i in range(len(train_list)):
        text.append(train_list[i])
        idx.append(str(index) + ' train ' + str(train_label[i]))
        index += 1
    for i in range(len(dev_list)):
        text.append(dev_list[i])
        idx.append(str(index) + ' dev ' + str(dev_label[i]))
        index += 1
    for i in range(len(test_list)):
        text.append(test_list[i])
        idx.append(str(index) + ' test ' + str(test_label[i]))
        index += 1
    json.dump(text, open('data/ours_query_dict_0.json', 'w'))
    json.dump(idx, open('data/split_label_ours_0.json', 'w'))
    
    
def gen_rawtext_iflytekfew():
    data = load_dataset('fewclue', 'iflytek')
    label_dict = {}
    for item in data['train_few_all']:
        if item['label_des'] not in label_dict:
            label_dict[item['label_des']] = len(label_dict)
    train_list = []
    test_list = []
    dev_list = []
    train_label = []
    test_label = []
    dev_label = []
    stopwords = load_stopwords()
    for i in tqdm(range(len(data['train_few_all']))):
        seg = jn.extract_tags(data['train_few_all'][i]['sentence'], topK=6, allowPOS=('n', 'ns', 'nt', 'vn', 'nw'))
        train_list.append(seg)
        train_label.append(label_dict[data['train_few_all'][i]['label_des']])
    for i in tqdm(range(len(data['test_public']))):
        seg = jn.extract_tags(data['test_public'][i]['sentence'], topK=6, allowPOS=('n', 'ns', 'nt', 'vn', 'nw'))
        test_list.append(seg)
        test_label.append(label_dict[data['test_public'][i]['label_des']])
    for i in tqdm(range(len(data['dev_few_all']))):
        seg = jn.extract_tags(data['dev_few_all'][i]['sentence'], topK=6, allowPOS=('n', 'ns', 'nt', 'vn', 'nw'))
        dev_list.append(seg)
        dev_label.append(label_dict[data['dev_few_all'][i]['label_des']])
    text = []
    idx = []
    index = 0
    for i in range(len(train_list)):
        text.append(train_list[i])
        idx.append(str(index) + ' train ' + str(train_label[i]))
        index += 1
    for i in range(len(test_list)):
        text.append(test_list[i])
        idx.append(str(index) + ' test ' + str(test_label[i]))
        index += 1
    for i in range(len(dev_list)):
        text.append(dev_list[i])
        idx.append(str(index) + ' dev ' + str(dev_label[i]))
        index += 1
    json.dump(text, open('../data_processed/EG/iflytek_query_dict_0.json', 'w'))
    json.dump(idx, open('../data_processed/EG/split_label_iflytek_0.json', 'w'))


def gen_rawtext_eprstmt():
    data = load_dataset('fewclue', 'eprstmt')
    label_dict = {}
    for item in data['train_few_all']:
        if item['label'] not in label_dict:
            label_dict[item['label']] = len(label_dict)
    train_list = []
    test_list = []
    dev_list = []
    train_label = []
    test_label = []
    dev_label = []
    for i in tqdm(range(len(data['train_few_all']))):
        seg = jn.extract_tags(data['train_few_all'][i]['sentence'], topK=6, allowPOS=('n', 'ns', 'nt', 'vn', 'nw'))
        train_list.append(seg)
        train_label.append(label_dict[data['train_few_all'][i]['label']])
    for i in tqdm(range(len(data['test_public']))):
        seg = jn.extract_tags(data['test_public'][i]['sentence'], topK=6, allowPOS=('n', 'ns', 'nt', 'vn', 'nw'))
        test_list.append(seg)
        test_label.append(label_dict[data['test_public'][i]['label']])
    for i in tqdm(range(len(data['dev_few_all']))):
        seg = jn.extract_tags(data['dev_few_all'][i]['sentence'], topK=6, allowPOS=('n', 'ns', 'nt', 'vn', 'nw'))
        dev_list.append(seg)
        dev_label.append(label_dict[data['dev_few_all'][i]['label']])
    text = []
    idx = []
    index = 0
    for i in range(len(train_list)):
        text.append(train_list[i])
        idx.append(str(index) + ' train ' + str(train_label[i]))
        index += 1
    for i in range(len(test_list)):
        text.append(test_list[i])
        idx.append(str(index) + ' test ' + str(test_label[i]))
        index += 1
    for i in range(len(dev_list)):
        text.append(dev_list[i])
        idx.append(str(index) + ' dev ' + str(dev_label[i]))
        index += 1
    json.dump(text, open('../data_processed/EG/eprstmt_query_dict_0.json', 'w'))
    json.dump(idx, open('../data_processed/EG/split_label_eprstmt_0.json', 'w'))
    
    
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


def generate_text_sample():
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
        text = []
        idx = []
        index = 0
        for j in range(500):
            seg = jieba.lcut(train_list[j], cut_all=False)
            cs = [c for c in seg if c not in stopwords and len(c) > 1 and is_number(c) == 0]
            text.append(cs)
            idx.append(str(index) + ' train ' + str(label_dict[train_label[j]]))
            index += 1
        for j in range(500,1000):
            seg = jieba.lcut(train_list[j], cut_all=False)
            cs = [c for c in seg if c not in stopwords and len(c) > 1 and is_number(c) == 0]
            text.append(cs)
            idx.append(str(index) + ' test ' + str(label_dict[train_label[j]]))
            index += 1
        for j in range(1000, 2000):
            seg = jieba.lcut(train_list[j], cut_all=False)
            cs = [c for c in seg if c not in stopwords and len(c) > 1 and is_number(c) == 0]
            text.append(cs)
            idx.append(str(index) + ' dev ' + str(label_dict[train_label[j]]))
            index += 1
        json.dump(text, open('../data_processed/EG/sample_query_dict_{}.json'.format(i), 'w'))
        json.dump(idx, open('../data_processed/EG/split_label_sample_{}.json'.format(i), 'w'))


# 使用模型进行预测
def generate_predict(predict_data):
    train_list = []
    test_list = []
    dev_list = []
    stopwords = load_stopwords()
    sen = predict_data[0]
    seg = jn.extract_tags(sen, topK=10, allowPOS=('n', 'ns', 'nt', 'vn', 'nw','nz','nr'))
    cs = [c for c in seg if c not in stopwords and len(c) > 1 and is_number(c) == 0]
    train_list.append(cs)
    for i in tqdm(range(len(predict_data))):
        sen = predict_data[i]
        seg = jn.extract_tags(sen, topK=10, allowPOS=('n', 'ns', 'nt', 'vn', 'nw', 'nz', 'nr'))
        cs = [c for c in seg if c not in stopwords and len(c) > 1 and is_number(c) == 0]
        test_list.append(cs)
    sen = predict_data[1]
    seg = jn.extract_tags(sen, topK=10, allowPOS=('n', 'ns', 'nt', 'vn', 'nw', 'nz', 'nr'))
    cs = [c for c in seg if c not in stopwords and len(c) > 1 and is_number(c) == 0]
    dev_list.append(cs)
    text = []
    idx = []
    index = 0
    for i in range(len(train_list)):
        text.append(train_list[i])
        idx.append(str(index) + ' train ' + str(1))
        index += 1
    for i in range(len(dev_list)):
        text.append(dev_list[i])
        idx.append(str(index) + ' dev ' + str(1))
        index += 1
    for i in range(len(test_list)):
        text.append(test_list[i])
        idx.append(str(index) + ' test ' + str(1))
        index += 1
    json.dump(text, open('data/predict_query_dict_0.json', 'w'))
    json.dump(idx, open('data/split_label_predict_0.json', 'w'))


def generate_text_byLabel(dataset_name):
    dataset = load_dataset(dataset_name)
    label_dic = dataset['train'].label_list
    print(label_dic)
    train_dict = {}
    test_dict = {}
    dev_dict = {}
    for la in label_dic:
        train_dict[la] = []
        test_dict[la] = []
        dev_dict[la] = []
    doc_list = []
    d = tqdm(dataset['train'])
    for line in d:
        train_dict[label_dic[line['label']]].append(
            jn.extract_tags(line['text'], topK=6, allowPOS=('n', 'ns', 'nt', 'vn', 'nw')))
        # doc_list.append(line['text'])
    d = tqdm(dataset['test'])
    for line in d:
        test_dict[label_dic[line['label']]].append(
            jn.extract_tags(line['text'], topK=6, allowPOS=('n', 'ns', 'nt', 'vn', 'nw')))
        # doc_list.append(line['text'])
    d = tqdm(dataset['dev'])
    for line in d:
        dev_dict[label_dic[line['label']]].append(
            jn.extract_tags(line['text'], topK=6, allowPOS=('n', 'ns', 'nt', 'vn', 'nw')))
        # doc_list.append(line['text'])
    doc_list.append(train_dict)
    doc_list.append(test_dict)
    doc_list.append(dev_dict)
    keyword = []
    # for line in tqdm(doc_list):
    #     keyword.append(jn.textrank(line,topK=5,allowPOS=('n','ns','nt','vn','nw')))
    # # print(keyword)
    json.dump(doc_list, open('../data_processed/EG/text_list_byLabel_{}.json'.format(dataset_name), 'w'))


def gen_label_text(dataset):
    doc_list = json.load(open('../data_processed/EG/text_list_byLabel_{}.json'.format(dataset), 'r'))
    label_dic = {}
    # ld={}
    # for la in label_dic:
    #     ld[la]=len(ld)
    train_idx = []
    train_il = []
    train_text = []
    for i in range(5):
        train_text.append([])
    for i in range(5):
        train_idx.append([])
    train_tl = []
    for key in doc_list[0].keys():
        if key not in label_dic:
            label_dic[key] = len(label_dic)
    index = 0
    idd = 0
    for t in tqdm(range(500)):
        if t % 100 == 0 and t != 0:
            index += 1
            idd = 0

        for key, value in doc_list[0].items():
            flag = False
            for i in range(t * 10, len(value)):
                if i % 10 == 0 and flag:
                    break
                # print(value[i])
                train_text[index].append(value[i])
                train_idx[index].append(str(idd) + ' train ' + str(label_dic[key]))
                flag = True
                idd += 1
    index = 0
    idd = 0
    for t in tqdm(range(100)):

        if t % 20 == 0 and t != 0:
            index += 1
            idd = 0

        for key, value in doc_list[1].items():
            flag = False
            for i in range(t * 10, len(value)):
                if i % 10 == 0 and flag:
                    break
                train_text[index].append(value[i])
                train_idx[index].append(str(idd + 5000) + ' test ' + str(label_dic[key]))
                flag = True
                idd += 1
    index = 0
    idd = 0
    for t in tqdm(range(50)):
        if t % 10 == 0 and t != 0:
            index += 1
            idd = 0
        for key, value in doc_list[2].items():
            flag = False
            for i in range(t * 10, len(value)):
                if i % 10 == 0 and flag:
                    break
                train_text[index].append(value[i])
                train_idx[index].append(str(idd + 6000) + ' dev ' + str(label_dic[key]))
                flag = True
                idd += 1
    for i in range(len(train_text)):
        json.dump(train_text[i], open('../data_processed/EG/{}_query_dict_{}.json'.format(dataset, i), 'w'))
        json.dump(train_idx[i], open('../data_processed/EG/split_label_{}_{}.json'.format(dataset, i), 'w'))


# 生成mapping
def process_raw_text(dataset, method):
    if method == 'entity':
        doc_list = json.load(open('../data_processed/EG/entity_list_{}.json'.format(dataset), 'r'))
    elif method == 'textrank':
        doc_list = json.load(open('../data_processed/EG/text_list_{}.json'.format(dataset), 'r'))
    else:
        print('选择一种文本生成方法！')
        return
    print(len(doc_list))
    query_dict = {}
    for words in tqdm(doc_list):
        for word in words:
            if word not in query_dict:
                query_dict[word] = len(query_dict)
    print(len(query_dict))
    json.dump(query_dict, open('../data_processed/EG/{}_query_dict.json'.format(dataset), 'w'))


# 获取标签列表
def get_label_split(dataset_name):
    dataset = load_dataset(dataset_name)
    label_list = []
    label_dic = dataset['train'].label_list
    print(dataset['train'][0])
    i = 0
    for line in dataset['train']:
        label_list.append(str(i) + ' train ' + str(line['label']))
        i += 1
    for line in dataset['test']:
        label_list.append(str(i) + ' test ' + str(line['label']))
        i += 1
    for line in dataset['dev']:
        label_list.append(str(i) + ' dev ' + str(line['label']))
        i += 1
    json.dump(label_list, open('../data_processed/EG/split_label_{}.json'.format(dataset_name), 'w'))


# split输入每一行是 'index 训练/测试/验证 类别'
def process_adj(dataset, i, eval=False, wiki=False):
    train_label_map, valid_label_map, test_label_map = {}, {}, {}
    label_list = []
    if eval:
        query_dict = json.load(open('data/{}_query_dict_{}.json'.format(dataset, str(i)), 'r'))
        split_label_list = json.load(open('data/split_label_{}_{}.json'.format(dataset, str(i)), 'r'))
        labels = []
        train_idx, valid_idx, test_idx = [], [], []
        word_list = []
        for line in split_label_list:
            temp = line.split(' ')
            text_tmp = query_dict[int(temp[0])]
            labels.append(int(temp[2]))
            if temp[1] == 'train':
                train_idx.append(int(temp[0]))
            elif temp[1] == 'test':
                test_idx.append(int(temp[0]))
            elif temp[1] == 'dev':
                valid_idx.append(int(temp[0]))
            WORD = text_tmp
            word_list.append(' '.join(WORD))
        word_mapping = json.load(open('data/ours_word_mapping.json'.format(dataset), 'r'))
        print(len(word_mapping))
        print('Length of [trian, valid, test, total]:', [len(train_idx), len(valid_idx), len(test_idx), len(labels)])
        adj_query2word = tf_idf_out_pool(word_list, word_mapping, sparse=False)
        joblib.dump(adj_query2word,
                    open('data/adj_query2word_{}_{}.joblib'.format(dataset, str(i)), 'wb'))
        json.dump([train_idx, valid_idx, test_idx],
                  open('data/text_index_{}_{}.json'.format(dataset, str(i)), 'w'))
        json.dump(labels, open('data/labels_{}_{}.json'.format(dataset, str(i)), 'w'))
        return
    if wiki:
        query_dict = json.load(open('../data_processed/EG/{}_query_dict_{}.json'.format(dataset, str(i)), 'r'))
        split_label_list = json.load(open('../data_processed/EG/split_label_{}_{}.json'.format(dataset, str(i)), 'r'))
        labels = []
        train_idx, valid_idx, test_idx = [], [], []
        word_list = []
        for line in split_label_list:
            temp = line.split(' ')
            text_tmp = query_dict[int(temp[0])]
            labels.append(int(temp[2]))
            if temp[1] == 'train':
                train_idx.append(int(temp[0]))
            elif temp[1] == 'test':
                test_idx.append(int(temp[0]))
            elif temp[1] == 'dev':
                valid_idx.append(int(temp[0]))
            WORD = text_tmp
            word_list.append(' '.join(WORD))
        word_mapping = json.load(open('../data/wiki_word_mapping.json', 'r'))
        print(len(word_mapping))
        print('Length of [trian, valid, test, total]:', [len(train_idx), len(valid_idx), len(test_idx), len(labels)])
        adj_query2word = tf_idf_out_pool(word_list, word_mapping, sparse=False)
        joblib.dump(adj_query2word,
                    open('../data_processed/EG/adj_query2word_{}_{}.joblib'.format(dataset, str(i)), 'wb'))
        json.dump([train_idx, valid_idx, test_idx],
                  open('../data_processed/EG/text_index_{}_{}.json'.format(dataset, str(i)), 'w'))
        json.dump(labels, open('../data_processed/EG/labels_{}_{}.json'.format(dataset, str(i)), 'w'))
        return

    query_dict = json.load(open('../data_processed/EG/{}_query_dict_{}.json'.format(dataset, str(i)), 'r'))
    split_label_list = json.load(open('../data_processed/EG/split_label_{}_{}.json'.format(dataset, str(i)), 'r'))
    labels = []
    train_idx, valid_idx, test_idx = [], [], []
    word_list = []
    for line in split_label_list:
        temp = line.split(' ')
        text_tmp = query_dict[int(temp[0])]
        labels.append(int(temp[2]))
        if temp[1] == 'train':
            train_idx.append(int(temp[0]))
        elif temp[1] == 'test':
            test_idx.append(int(temp[0]))
        elif temp[1] == 'dev':
            valid_idx.append(int(temp[0]))
        WORD = text_tmp
        word_list.append(' '.join(WORD))
    word_mapping = json.load(open('../data_processed/EWG/{}_word_mapping.json'.format(dataset), 'r'))
    print(len(word_mapping))
    print('Length of [trian, valid, test, total]:', [len(train_idx), len(valid_idx), len(test_idx), len(labels)])
    adj_query2word = tf_idf_out_pool(word_list, word_mapping, sparse=False)
    joblib.dump(adj_query2word, open('../data_processed/EG/adj_query2word_{}_{}.joblib'.format(dataset, str(i)), 'wb'))
    json.dump([train_idx, valid_idx, test_idx],
              open('../data_processed/EG/text_index_{}_{}.json'.format(dataset, str(i)), 'w'))
    json.dump(labels, open('../data_processed/EG/labels_{}_{}.json'.format(dataset, str(i)), 'w'))


if __name__ == '__main__':
    # entity_list=json.load(open('../data_processed/EG/entity_list.json','r'))
    # word_map=json.load(open('../data_processed/EWG/thucnews_word_mapping.json','r'))
    # print(len((word_map)))
    # wm=set()
    # find=[]
    # for key in word_map.keys():
    #     wm.add(key)
    # for line in entity_list:
    #     for enti in line:
    #         if enti in wm:
    #             find.append(enti)
    # print(len(find))
    # find=set(find)
    # print(len(find))
    # print(find)
    # get_label_split('thucnews')
    # pool_doc_list = json.load(open('../data_processed/EWG/thucnews_word_list.json', 'r'))
    # print(pool_doc_list[0])
    # generate_text('thucnews')
    # process_raw_text('thucnews','textrank')
    # get_label_split_red('thucnews')
    # # process_adj('thucnews')
    # adj=joblib.load(open('../data_processed/EG/adj_query2word_thucnews.joblib', 'rb'))
    # print(adj.shape)
    # print(adj[0])
    # generate_text_byLabel('thucnews')
    # gen_label_text('thucnews')
    # gen_rawtext_tnews(eval=0)
    generate_text_ours()
    # gen_rawtext_csldcp()
    # generate_text_sample()
    # gen_thucnews_short(0)
    # gen_rawtext_tnewsfew()
    process_adj('ours', i=0, eval=0, wiki=0)