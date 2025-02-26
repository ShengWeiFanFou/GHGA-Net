import os
import time

from tqdm import tqdm


# 从文档中按段落抽取文本
def line_text_gen(path=r'C:\Users\byz\Desktop\开题\美国国家档案馆数据集\数据集\dataset'):
    train_list=[]
    dev_list=[]
    test_list=[]
    label_dict={}
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        print(file)
        label=file
        if label not in label_dict:
            label_dict[label]=len(label_dict)
        if os.path.isdir(file_path):
            num = 0
            for doc in os.listdir(file_path):
                doc_path=os.path.join(file_path, doc)
                with open(doc_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    legt=int(len(lines)/4)
                    for i in range(len(lines)):
                        if len(lines[i])<5:
                            continue
                        # l=lines[i].strip('\n') + '\t' + str(label_dict[label])+'\n'
                        # print(l)
                        if legt*2<i<=legt*3:
                            # dev_list.append(l)
                            test_list.append(label + '\t' + lines[i].strip('\n'))
                        elif i>legt*3:
                            # test_list.append(l)
                            dev_list.append(label + '\t' + lines[i].strip('\n'))
                        else:
                            # train_list.append(l)
                            train_list.append(label + '\t' + lines[i].strip('\n'))
                        num+=1
            print(label+"条数为:"+str(num))
    with open('Qwen_Gen.txt','r',encoding='utf-8')as qw:
        ql=qw.readlines()
        qlen=len(ql)
        for i in range(qlen):
            if i<=(qlen/3):
                train_list.append(ql[i])
            elif (qlen/3)<i<(2*qlen/3):
                test_list.append(ql[i])
            else:
                dev_list.append(ql[i])
    print(len(train_list))
    print(len(test_list))
    print(len(dev_list))
    return train_list,test_list,dev_list,label_dict


def line_text_gen_sum():
    train_list=[]
    dev_list=[]
    test_list=[]
    label_dict = {'unclassified': '0', 'confidential': '1', 'secret': '2', 'topsecret': '3'}
    with open('summary/trainSummary.txt','r',encoding='utf-8')as t:
        lines=t.readlines()
        for l in lines:
            train_list.append(l)
    with open('summary/testSummary.txt','r',encoding='utf-8')as te:
        lines=te.readlines()
        for l in lines:
            test_list.append(l)
    with open('summary/devSummary.txt','r',encoding='utf-8')as d:
        lines=d.readlines()
        for l in lines:
            dev_list.append(l)
    return train_list,test_list,dev_list,label_dict


def line_text_gen_txt():
    train_list=[]
    dev_list=[]
    test_list=[]
    label_dict = {'unclassified': '0', 'confidential': '1', 'secret': '2', 'topsecret': '3'}
    with open('train.txt','r',encoding='utf-8')as t:
        lines=t.readlines()
        for l in lines:
            train_list.append(l)
    with open('test.txt','r',encoding='utf-8')as te:
        lines=te.readlines()
        for l in lines:
            test_list.append(l)
    with open('dev.txt','r',encoding='utf-8')as d:
        lines=d.readlines()
        for l in lines:
            dev_list.append(l)
    return train_list,test_list,dev_list,label_dict


def line_text_gen_txt_for_other():
    train_list=[]
    dev_list=[]
    test_list=[]
    label_dict = {'unclassified': '0', 'confidential': '1', 'secret': '2', 'topsecret': '3'}
    with open('train.txt','r',encoding='utf-8')as t:
        lines=t.readlines()
        for l in lines:
            train_list.append(l)
    with open('test.txt','r',encoding='utf-8')as te:
        lines=te.readlines()
        for l in lines:
            test_list.append(l)
    with open('dev.txt','r',encoding='utf-8')as d:
        lines=d.readlines()
        for l in lines:
            dev_list.append(l)
    with open('trainO.txt','w',encoding='utf-8')as t:
        for tl in train_list:
            label,text=tl.strip('\n').split('\t')
            t.write(text+'\t'+label_dict[label]+'\n')
    with open('testO.txt','w',encoding='utf-8')as t:
        for tl in test_list:
            label,text=tl.strip('\n').split('\t')
            t.write(text+'\t'+label_dict[label]+'\n')
    with open('devO.txt','w',encoding='utf-8')as t:
        for tl in dev_list:
            label,text=tl.strip('\n').split('\t')
            t.write(text+'\t'+label_dict[label]+'\n')


def line_text_gen_for_other_model(path=r'C:\Users\byz\Desktop\开题\美国国家档案馆数据集\数据集\dataset'):
    train_list=[]
    dev_list=[]
    test_list=[]
    label_dict={}
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        print(file)
        label=file
        if label not in label_dict:
            label_dict[label]=len(label_dict)
        if os.path.isdir(file_path):
            num = 0
            for doc in os.listdir(file_path):
                doc_path=os.path.join(file_path, doc)
                with open(doc_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    legt=int(len(lines)/4)
                    for i in range(len(lines)):
                        if len(lines[i])<5:
                            continue
                        # l=lines[i].strip('\n') + '\t' + str(label_dict[label])+'\n'
                        # print(l)
                        if legt*2<i<=legt*3:
                            # dev_list.append(l)
                            # train_list.append(lines[i].strip("\n")+ '\t'+str(label_dict[label])+'\n')
                            test_list.append(label + '\t' + lines[i].strip("\n")+'\n')
                        elif i>legt*3:
                            # test_list.append(l)
                            # dev_list.append(lines[i].strip("\n")+ '\t'+str(label_dict[label])+'\n')
                            dev_list.append(label + '\t' + lines[i].strip("\n")+'\n')
                        else:
                            # train_list.append(l)
                            # test_list.append(lines[i].strip("\n")+ '\t'+str(label_dict[label])+'\n')
                            train_list.append(label + '\t' + lines[i].strip("\n")+'\n')
                        num+=1
            print(label+"条数为:"+str(num))
    with open('Qwen_Gen.txt','r',encoding='utf-8')as qw:
        ql=qw.readlines()
        qlen=len(ql)
        for i in tqdm(range(qlen)):
            lb,se=ql[i].split('\t')
            if i<=(qlen/3):
                train_list.append(ql[i])
            elif (qlen/3)<i<(2*qlen/3):
                test_list.append(ql[i])
            else:
                dev_list.append(ql[i])
    print(len(train_list))
    print(len(test_list))
    print(len(dev_list))
    return train_list,test_list,dev_list,label_dict


def entire_text_gen(path=r'C:\Users\byz\Desktop\开题\美国国家档案馆数据集\数据集\dataset'):
    doc_list=[]
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        print(file)
        label=file
        if os.path.isdir(file_path):
            for doc in os.listdir(file_path):
                doc_path=os.path.join(file_path, doc)
                with open(doc_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines:
                        if len(line)>5:
                            doc_list.append(line)
    return doc_list


def entire_text_gen_txt():
    train_list=[]
    dev_list=[]
    test_list=[]
    with open('train.txt','r',encoding='utf-8')as t:
        lines=t.readlines()
        for l in lines:
            train_list.append(l)
    with open('test.txt','r',encoding='utf-8')as te:
        lines=te.readlines()
        for l in lines:
            test_list.append(l)
    with open('dev.txt','r',encoding='utf-8')as d:
        lines=d.readlines()
        for l in lines:
            dev_list.append(l)
    return train_list+test_list+dev_list


def entire_text_gen_sum(path=r'D:\bishesystem\GHGC-Net_pkg\summary'):
    doc_list=[]
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if len(line) > 5:
                    doc_list.append(line)
    return doc_list


if __name__ == '__main__':
    # train_list, test_list, dev_list, label_dict = line_text_gen()
    # train_list,test_list,dev_list,label_dict=line_text_gen_for_other_model()
    # with open('train.txt','w',encoding='utf-8')as train:
    #     train.writelines(train_list)
    # with open('test.txt','w',encoding='utf-8')as test:
    #     test.writelines(test_list)
    # with open('dev.txt','w',encoding='utf-8')as dev:
    #     dev.writelines(dev_list)

    # with open('./all.txt','w',encoding='utf-8')as all:
    #     all.writelines(train_list)
    #     all.writelines(test_list)
    #     all.writelines(dev_list)
    # with open('class.txt','w',encoding='utf-8')as clas:
    #     for key in label_dict.keys():
    #         clas.write(key+'\n')
    line_text_gen_txt_for_other()