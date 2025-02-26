import argparse
import json
import statistics
import time

import paddle
from data_pre.entire_word_graph_gen import cacluate_PMI, equip_embed, process_raw_data, read_dataset_tfidf

from main.trainer import Trainer, set_seed
from data_pre.character_graph_gen import cacluate_PMI_c, equip_embed_c, generate_rawchar_ours, generate_rawchar_predict, process_adj_c, process_raw_data_c, read_dataset_rc
from data_pre.text_graph_gen import generate_predict, generate_text_ours, process_adj


# 知识切片
def slide_window(str):
    list=[]
    index = 0
    current = 128
    timeout = 128
    while current < len(str):
        if current == index:
            timeout += 32
            if current + timeout >= len(str) - 1:
                list.append(str[index:])
                break
            else:
                current += timeout
        if str[current] == '。' or str[current] == '.':
            list.append(str[index:current])
            index = current + 1
            if current + 128 >= len(str) - 1:
                list.append(str[index:])
                break
            else:
                current += 128
        else:
            current -= 1
    return list


# 文档级预测 分段落
def data_for_predict(file_path):
    start=time.time()
    doc_list=[]
    with open(file_path,'r',encoding='gbk') as f:
        lines = f.read()
        doc_list=slide_window(lines)
    data = 'predict'
    # 全局文档词图构建
    read_dataset_tfidf(data,doc_list)
    process_raw_data(1, data)
    equip_embed(data)
    cacluate_PMI(data)
    generate_predict(doc_list)
    process_adj(data, i=0, eval=1, wiki=0)
    # 全局文档字图构建
    read_dataset_rc(data,doc_list)
    process_raw_data_c(10, data)
    equip_embed_c(data)
    cacluate_PMI_c(data)
    generate_rawchar_predict(doc_list)
    process_adj_c(data, i=0)
    print('数据预处理完成'+'耗时'+str(time.time()-start))


    parser = argparse.ArgumentParser(description="model params")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dataset", "-d", type=str, default='predict', help='thucnews tnews iflytek csldcp')
    parser.add_argument("--train_num", type=int, default=200)
    parser.add_argument("--file_dir", "-f_dir", type=str, default='./')
    parser.add_argument("--data_path", "-d_path", type=str, default='data/')
    parser.add_argument('--disable_cuda', action='store_true')
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--drop_out", type=float, default=0.6)
    parser.add_argument("--max_epoch", type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--trained', type=int, default=0)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--do_eval', type=bool, default=0)
    parser.add_argument('--predict', type=bool, default=1)
    params = parser.parse_args()
    # if not params.disable_cuda and paddle.device.get_device():
    #     params.device = paddle.device.set_device('gpu:%d' % params.gpu)
    # else:
    params.device = paddle.device.set_device('cpu')
    set_seed(params.seed)
    trainer = Trainer(params)
    # 敏感级别判断
    label,l=trainer.eval()
    type=['Confidential','Secret','Topsecret','Unclassified']
    flag=False
    if type[label]=='Confidential' or type[label]=='Secret' or type[label]=='Topsecret':
        print("检测到敏感文件！！！")
        flag=True
    print('该文件敏感级别为'+type[label])
    print('total time: ', time.time() - start)
    label_dict={'Confidential':0,'Secret':0,'Topsecret':0,'Unclassified':0}
    length=len(l)
    for i in l:
        label_dict[type[i]]+=1
    res=''
    for key,values in label_dict.items():
        per=float(values/length*100)
        res+=key+':'+str(per)+'% '
    return type[label],res,time.time() - start,flag

# 模型训练
def model_train(data):
    start=time.time()
    doc_list=[]
    read_dataset_tfidf(data,doc_list)
    process_raw_data(1, data,50)
    equip_embed(data)
    cacluate_PMI(data)
    generate_text_ours(data)
    process_adj(data, i=0, eval=1, wiki=0)
    read_dataset_rc(data,doc_list)
    process_raw_data_c(10, data)
    equip_embed_c(data)
    cacluate_PMI_c(data)
    generate_rawchar_ours(data)
    process_adj_c(data, i=0)
    print('数据预处理完成'+'耗时'+str(time.time()-start))


    parser = argparse.ArgumentParser(description="model params")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dataset", "-d", type=str, default=data, help='thucnews tnews iflytek csldcp')
    parser.add_argument("--train_num", type=int, default=200)
    parser.add_argument("--file_dir", "-f_dir", type=str, default='./')
    parser.add_argument("--data_path", "-d_path", type=str, default='data/')
    parser.add_argument('--disable_cuda', action='store_true')
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--drop_out", type=float, default=0.6)
    parser.add_argument("--max_epoch", type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--trained', type=int, default=0)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--do_eval', type=bool, default=0)
    parser.add_argument('--predict', type=bool, default=0)
    params = parser.parse_args()
    if not params.disable_cuda and paddle.device.get_device():
        params.device = paddle.device.set_device('gpu:%d' % params.gpu)
    else:
        params.device = paddle.device.set_device('cpu')

    set_seed(params.seed)
    trainer = Trainer(params)
    trainer.train()
    del trainer
    print('total time: ', time.time() - start)
    # label=trainer.eval()
    # type=['confidential','secret','topsecret','unclassified']
    # if type[label]=='confidential' or type[label]=='secret' or type[label]=='topsecret':
    #     print("检测到涉密文件！！！")
    # print('该文件密级为'+type[label])

# 融合决策
def fusionPredict():
    # 读取语句级分类器结果
    ghga=json.load(open('GHGA-Net_predict.json','r'))
    # 读取篇章级分类器结果
    ernie=json.load(open('ERNIE3-Sens_predict.json','r'))
    # 保存加权结果
    pred_l=[]
    for i in range(len(ghga)):
        pred=[]
        for j in range(len(ghga[i])):
            # 线性插值
            final_score=0.5*ernie[i][j]+0.5*ghga[i][j]
            pred.append(final_score)
        pred_l.append(pred)
    rel=[]
    for re in pred_l:
        label=statistics.mode(re)
        rel.append(label)
    return  statistics.mode(rel)


if __name__ == '__main__':
    path=input('请给出文件地址：')
    ifpath=input('是否为windows复制的路径名？ y/n ')
    if ifpath=='y':
        path=path.strip('"')
        p=path.split('\\')
        path='/'.join(p)
    data_for_predict(path)
    # model_train('ours')