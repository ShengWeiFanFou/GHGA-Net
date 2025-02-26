import os
import random

from tqdm import tqdm


def read_txt_short():
    text_list=[]
    with open('train.txt','r',encoding='utf-8')as t:
        lines=t.readlines()
        for l in lines:
            text_list.append(l)
    with open('test.txt','r',encoding='utf-8')as te:
        lines=te.readlines()
        for l in lines:
            text_list.append(l)
    with open('dev.txt','r',encoding='utf-8')as d:
        lines=d.readlines()
        for l in lines:
            text_list.append(l)
    print(len(text_list))
    long=0
    longlist=[]
    shortlist = []
    for text in text_list:
        te=text.strip('\n').split('\t')[1]
        if(len(text)>50):
            long+=1
            longlist.append(text)
        else:
            shortlist.append(text.strip('\n'))
    print(long)
    for lt in longlist:
        shortlist=genSTC(lt,shortlist)
    print(len(shortlist))
    with open('shortSTC.txt','w',encoding='utf-8')as d:
        for st in shortlist:
            if len(st)>13:
                d.write(st+'\n')


def read_text_long(path=r'C:\Users\byz\Desktop\开题\美国国家档案馆数据集\数据集\dataset'):
    train_list=[]
    label_dict={}
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        print(file)
        label=file
        if label not in label_dict:
            label_dict[label]=len(label_dict)
        if os.path.isdir(file_path):
            for doc in tqdm(os.listdir(file_path)):
                doc_path=os.path.join(file_path, doc)
                print(doc)
                with open(doc_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    totaltext = ''
                    for line in lines:
                        totaltext+=line.strip('\n')

                train_list=genDoc(totaltext,train_list,label)
            print(len(train_list))
    # with open('Qwen_Gen.txt','r',encoding='utf-8')as qw:
    #     ql=qw.readlines()
    #     qlen=len(ql)
    #     for i in range(qlen):
    #         if i<=(qlen/3):
    #             train_list.append(ql[i])
    #         elif (qlen/3)<i<(2*qlen/3):
    #             test_list.append(ql[i])
    #         else:
    #             dev_list.append(ql[i])
    print(len(train_list))
    with open('docC.txt', 'w', encoding='utf-8') as d:
        for st in train_list:
            d.write(st + '\n')


def genLong(str,list,label):
    index=0
    current=256
    timeout=256
    while current<len(str):
        if current==index:
            timeout+=32
            if current + timeout >= len(str) - 1:
                list.append(label + '\t' + str[index:])
                break
            else:
                current += timeout
        if str[current]=='。' or str[current]=='.':
            list.append(label+'\t'+str[index:current])
            index=current+1
            if current +256>= len(str)-1 :
                list.append(label + '\t' + str[index:])
                break
            else:
                current+=256
        else:
            current-=1
    return list


def genDoc(str,list,label):
    index=0
    current=1536
    timeout=1536
    while current<len(str):
        if current==index:
            timeout+=32
            if current + timeout >= len(str) - 1:
                list.append(label + '\t' + str[index:])
                break
            else:
                current += timeout
        if str[current]=='。' or str[current]=='.':
            list.append(label+'\t'+str[index:current])
            index=current+1
            if current +1536>= len(str)-1 :
                list.append(label + '\t' + str[index:])
                break
            else:
                current+=1536
        else:
            current-=1
    return list


def genSTC(lt,shortlist):
    label,str=lt.strip("\n").split('\t')
    index=0
    for i in range(len(str)):
        if i==len(str)-1:
            shortlist.append(label+'\t'+str[index:i])
            break
        if str[i]=='。':
            shortlist.append(label+'\t'+str[index:i])
            index=i+1
    return shortlist


def conclu():
    label_dict = {'unclassified': 0, 'confidential': 0, 'secret': 0, 'topsecret': 0}
    sl=0
    with open('shortSTC.txt','r',encoding='utf-8')as d:
        shortlist=d.readlines()
    for st in shortlist:
        label,str=st.strip("\n").split('\t')
        label_dict[label]+=1
        sl+=len(str)
    print("CIA-Sens-CN-Short数据集")
    print("数据分布")
    print(label_dict)
    print("总数据条数")
    print(len(shortlist))
    print("平均长度")
    print(sl/len(shortlist))
    ll=0
    label_dict_2 = {'unclassified': 0, 'confidential': 0, 'secret': 0, 'topsecret': 0}
    with open('longTC.txt','r',encoding='utf-8')as d:
        longlist=d.readlines()
    for lt in longlist:
        label,str=lt.strip("\n").split('\t')
        label_dict_2[label]+=1
        ll+=len(str)
    print("CIA-Sens-CN-Long数据集")
    print("数据分布")
    print(label_dict_2)
    print("总数据条数")
    print(len(longlist))
    print("平均长度")
    print(ll/len(longlist))
    dl=0
    label_dict_3 = {'unclassified': 0, 'confidential': 0, 'secret': 0, 'topsecret': 0}
    with open('docC.txt','r',encoding='utf-8')as d:
        doclist=d.readlines()
    for lt in doclist:
        label,str=lt.strip("\n").split('\t')
        label_dict_3[label]+=1
        dl+=len(str)
    print("CIA-Sens-CN-Doc数据集")
    print("数据分布")
    print(label_dict_3)
    print("总数据条数")
    print(len(doclist))
    print("平均长度")
    print(dl/len(doclist))


def trainShortGen():
    label_dict = {'unclassified': [], 'confidential': [], 'secret': [], 'topsecret': []}
    with open('shortSTC.txt','r',encoding='utf-8')as d:
        shortlist=d.readlines()
        for st in shortlist:
            label, str = st.strip("\n").split('\t')
            label_dict[label].append(st)
    trainl=[]
    testl=[]
    devl=[]
    for l in label_dict.values():
        random.shuffle(l)
        les=len(l)
        for i in range(les):
            if i<les/10:
                devl.append(l[i])
            elif les/10<=i<3*les/10:
                testl.append(l[i])
            elif 3*les / 10 <= i < 5 * les / 10:
                trainl.append(l[i])
    print(len(trainl))
    print(len(testl))
    print(len(devl))
    random.shuffle(trainl)
    random.shuffle(testl)
    random.shuffle(devl)
    with open('GHGA-train.txt','w',encoding='utf-8')as t:
        for tl in trainl:
            t.write(tl)
    with open('GHGA-test.txt','w',encoding='utf-8')as t:
        for tl in testl:
            t.write(tl)
    with open('GHGA-Dev.txt','w',encoding='utf-8')as t:
        for tl in devl:
            t.write(tl)
    # label_dict2 = {'unclassified': '0', 'confidential': '1', 'secret': '2', 'topsecret': '3'}
    # with open('CIA_DOC_TRAIN1.txt','w',encoding='utf-8')as t:
    #     for tl in trainl:
    #         label, str = tl.strip("\n").split('\t')
    #         t.write(str+'\t'+label_dict2[label]+'\n')
    # with open('CIA_DOC_TEST1.txt','w',encoding='utf-8')as t:
    #     for tl in testl:
    #         label, str = tl.strip("\n").split('\t')
    #         t.write(str + '\t' + label_dict2[label] + '\n')
    # with open('CIA_DOC_DEV1.txt','w',encoding='utf-8')as t:
    #     for tl in devl:
    #         label, str = tl.strip("\n").split('\t')
    #         t.write(str + '\t' + label_dict2[label] + '\n')


if __name__ == '__main__':
    # read_text_long()
    # with open(r"C:\Users\byz\Desktop\开题\美国国家档案馆数据集\数据集\dataset\unclassified\csldcp_text.txt", 'r', encoding='utf-8') as d:
    #     lines = d.readlines()
    #     totaltext = ''
    #     for line in lines:
    #         totaltext += line.strip('\n')
    #     print(len(totaltext))
    #     train_list = genLong(totaltext, [], '1')
    # print(len(train_list))
    # for tl in train_list:
    #     print(len(tl))
    # read_txt()
    # conclu()
    # trainShortGen()
    print(random.uniform(66,67))