# 适用于64位windows系统的office文件提取


import imghdr
import os
import shutil
import struct
import threading
import time
import zipfile
from tkinter import filedialog
import tkinter as tk

import pdfplumber
import win32con
import win32gui
from win32com import client as wc

# 判断文件是否含有密码
f1 = False


def checkIfPasswordWord(fil):
    global fl
    fl = False
    ter = False

    def terminate():
        global fl
        while True:
            hwnd = win32gui.FindWindow(None, '密码')
            if hwnd != 0:
                print('有密码')
                win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
                fl = True
                break
            if ter:
                break

    t = threading.Thread(target=terminate)
    t.start()
    try:
        doc = wc.Dispatch("Word.Application")
        doc.Documents.Open(fil, ReadOnly=True)
    except:
        t.join()
        doc.Quit()
        None
    if t.is_alive():
        ter = True
        t.join()
    print(fl)
    return fl


def checkIfPasswordExcel(fil):
    global fl
    fl = False
    ter = False

    def terminate():
        global fl
        while True:
            hwnd = win32gui.FindWindow(None, '密码')
            if hwnd != 0:
                print('有密码')
                win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
                fl = True
                break
            if ter:
                break

    t = threading.Thread(target=terminate)
    t.start()
    try:
        xls = wc.Dispatch("Excel.Application")
        xls.Workbooks.Open(fil, ReadOnly=True)
    except:
        t.join()
        xls.Quit()
        None
    if t.is_alive():
        ter = True
        t.join()
    print(fl)
    return fl


def checkIfPasswordPowerPoint(fil):
    global fl
    fl = False
    ter = False

    def terminate():
        global fl
        while True:
            hwnd = win32gui.FindWindow(None, '密码')
            if hwnd != 0:
                print('有密码')
                win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
                fl = True
                break
            if ter:
                break

    t = threading.Thread(target=terminate)
    t.start()
    try:
        powerpoint = wc.Dispatch('PowerPoint.Application')
        powerpoint.Presentations.Open(fil)

    except:
        t.join()
        # powerpoint.Quit()
        None
    if t.is_alive():
        ter = True
        powerpoint.Quit()
        t.join()
    print(fl)
    return fl


# 将所有ppt文件使用powerpoint转换为pptx文件
def checkIfPPTX(path_original):
    i = 0
    j = 0
    for file in os.listdir(path_original):
        if file.endswith('.ppt'):
            i += 1
    for file in os.listdir(path_original):
        if file.endswith('.ppt'):
            in_file = os.path.abspath(path_original + "//" + file)
            print(in_file)
            fa = checkIfPasswordPowerPoint(in_file)
            print(fa)
            if fa:
                print("该文件含有密码")
            else:
                powerpoint = wc.gencache.EnsureDispatch('PowerPoint.Application')
                ppt = powerpoint.Presentations.Open(in_file)
                ppt.SaveAs(in_file[:-4] + '.pptx')
                ppt.Close()
                print('转换成功')
            time.sleep(1)
            j += 1
            if i == j:
                powerpoint.Quit()


# 将所有xls文件使用excel转换为xlsx文件
def checkIfXlsx(path_original):
    for file in os.listdir(path_original):
        if file.endswith('.xls'):
            out_name = file.replace("xls", r'xlsx')  # doc文件修改后缀名
            in_file = os.path.abspath(path_original + "//" + file)
            out_file = os.path.abspath(path_original + "//" + out_name)
            print(in_file)
            print(out_file)
            fa = checkIfPasswordExcel(in_file)
            print(fa)
            if fa:
                print("该文件含有密码")
            else:
                excel = wc.Dispatch('Excel.Application')
                xls = excel.Workbooks.Open(in_file)
                xls.SaveAs(out_file, 51)
                xls.Close()
                print('转换成功')
                excel.Quit()
                time.sleep(2)


# 将所有doc文件使用word转换为docx文件
def checkIfDocx(path_original):
    for file in os.listdir(path_original):
        if file.endswith('.doc'):
            out_name = file.replace("doc", r'docx')  # doc文件修改后缀名
            in_file = os.path.abspath(path_original + "//" + file)
            out_file = os.path.abspath(path_original + "//" + out_name)
            print(in_file)
            print(out_file)
            fa = checkIfPasswordWord(in_file)
            print(fa)
            if fa:
                print("该文件含有密码")
            else:
                word = wc.Dispatch('Word.Application')
                doc = word.Documents.Open(in_file)
                doc.SaveAs(out_file, 12, False, "", True, "", False, False, False, False)
                doc.Close()
                print('转换成功')
                word.Quit()
                time.sleep(1)

# pdf解析
def extractPDF(path_original):
    for file in os.listdir(path_original):
        if file.endswith('.pdf'):
            pdftext=[]
            file_path=path_original+'/'+file
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    pdftext.append(page.extract_text())
            nf=path_original+'/pdfText-'+file.split('.')[0]+'.txt'
            with open(nf,'w',encoding='utf-8') as pf:
                for text in pdftext:
                    pf.write(text)

# 解压docx文件（实际上就是zip）
def deZipWord(path_original):
    for file in os.listdir(path_original):
        if file.endswith('.docx'):
            in_file = os.path.abspath(path_original + "//" + file)
            f = zipfile.ZipFile(in_file, "r")
            print(f.namelist())
            out_file = path_original + "//" + "in" + "//" + file.split(".")[0]
            if not os.path.exists(out_file):
                os.makedirs(out_file)
            # os.makedirs(out_file)
            for name in f.namelist():
                print(name)
                if name == "word/document.xml":
                    f.extract(name, out_file)
                a = name.split("/")
                if len(a) > 1:
                    if a[0] == "word" and (a[1] == "embeddings" or a[1] == "media"):
                        f.extract(name, out_file)


# 从解压的对应文件中提取出需要的信息
def getAllFilesWord(path_o):
    inpath = os.path.abspath(path_o + "//" + "in")
    if not os.path.exists(inpath):
        os.makedirs(inpath)
    outPath = os.path.abspath(path_o + "//" + "out")
    if not os.path.exists(outPath):
        os.makedirs(outPath)
    file = os.listdir(inpath)
    for fi in file:
        fp = os.path.join(inpath, fi)
        if os.path.isdir(fp):
            print(fi)
            for root, dirs, files in os.walk(fp):
                for name in files:
                    na = os.path.join(root, name)
                    print(na)
                    # 图片
                    if na.endswith(".emf") or imghdr.what(na) != None:
                        op1 = outPath + "//" + fi + "//" + "image"
                        if not os.path.exists(op1):
                            os.makedirs(op1)
                        op1n = op1 + "//" + name
                        shutil.copyfile(na, op1n)
                    # 文本
                    if na.endswith(".xml"):
                        text = ""
                        with open(na, "r", encoding="utf-8") as bf:
                            str = bf.readline()
                            print(str)
                            str = bf.readline()
                            aa = str.split("<w:t>")
                            aa = aa[1:]
                            for a in aa:
                                aaa = a.split("</w:t>")
                                text += aaa[0]
                            bf.close()
                        op2 = outPath + "//" + fi + "//" + "text"
                        if not os.path.exists(op2):
                            os.makedirs(op2)
                        op2n = op2 + "//" + fi + ".txt"
                        with open(op2n, "w") as newText:
                            newText.write(text)
                            newText.close()
                    # 内嵌文件
                    if na.endswith(".bin"):
                        op4 = outPath + "//" + fi + "//" + "office"
                        if not os.path.exists(op4):
                            os.makedirs(op4)
                        oleExtract(na, op4)
                    # 对应格式的内嵌文件
                    if na.endswith(".doc") or na.endswith(".docx") or na.endswith(".ppt") or na.endswith(
                            ".pptx") or na.endswith(".xls") or na.endswith(".xlsx"):
                        op3 = outPath + "//" + fi + "//" + "office"
                        if not os.path.exists(op3):
                            os.makedirs(op3)
                        op3n = op3 + "//" + name
                        shutil.copyfile(na, op3n)


# 解压xlsx文件（实际上就是zip）
def deZipExcel(path_original):
    for file in os.listdir(path_original):
        if file.endswith('.xlsx'):
            in_file = os.path.abspath(path_original + "//" + file)
            f = zipfile.ZipFile(in_file, "r")
            print(f.namelist())
            out_file = path_original + "//" + "in" + "//" + file.split(".")[0]
            if not os.path.exists(out_file):
                os.makedirs(out_file)
            # os.makedirs(out_file)
            for name in f.namelist():
                print(name)
                if name == "xl/sharedStrings.xml":
                    f.extract(name, out_file)
                if name == "xl/workbook.xml":
                    f.extract(name, out_file)
                a = name.split("/")
                if len(a) > 1:
                    if a[0] == "xl" and (a[1] == "embeddings" or a[1] == "media"):
                        f.extract(name, out_file)
                    if a[0] == "xl" and (a[1] == "worksheets" and a[2] != "_rels"):
                        f.extract(name, out_file)


# 从解压的对应文件中提取出需要的信息
def getAllFilesExcel(path_o):
    inpath = os.path.abspath(path_o + "//" + "in")
    if not os.path.exists(inpath):
        os.makedirs(inpath)
    outPath = os.path.abspath(path_o + "//" + "out")
    if not os.path.exists(outPath):
        os.makedirs(outPath)
    file = os.listdir(inpath)
    for fi in file:
        fp = os.path.join(inpath, fi)
        if os.path.isdir(fp):
            print(fi)
            sw = 0  # 为了判断sharedString和workbook均已读取完毕
            sheetC = 0
            sheetName = []
            sharedString = []
            op2 = outPath + "//" + fi + "//" + "text"
            if not os.path.exists(op2):
                os.makedirs(op2)
            op2n = op2 + "//" + fi + ".txt"
            for root, dirs, files in os.walk(fp):
                for name in files:
                    na = os.path.join(root, name)
                    print(na)
                    print(name)
                    # 图片
                    if na.endswith(".emf") or imghdr.what(na) != None:
                        op1 = outPath + "//" + fi + "//" + "image"
                        if not os.path.exists(op1):
                            os.makedirs(op1)
                        op1n = op1 + "//" + name
                        shutil.copyfile(na, op1n)
                    # 处理表格内容
                    # sharedString储存了表格中所有的字符串 构建一个列表进行存储
                    if na.endswith("sharedStrings.xml"):
                        with open(na, "r", encoding="utf-8") as ss:
                            str = ss.readline()
                            print(str)
                            str = ss.readline()
                            aa = str.split("<t>")
                            aa = aa[1:]
                            for a in aa:
                                aaa = a.split("</t>")
                                sharedString.append(aaa[0])
                            ss.close()
                        sw += 1
                    # workbook 获取不同表格的名称
                    if na.endswith("workbook.xml"):
                        with open(na, "r", encoding="utf-8") as ss:
                            str = ss.readline()
                            print(str)
                            str = ss.readline()
                            aa = str.split('<sheet name="')
                            aa = aa[1:]
                            for a in aa:
                                aaa = a.split('"')
                                sheetName.append(aaa[0])
                            ss.close()
                        sw += 1
                    # sheet 获取各个表格内容并与sharedString对应
                    sheet = []
                    lenT = 0
                    print(sw)
                    if root.endswith("worksheets") and sw == 2:
                        with open(na, "r", encoding="utf-8") as ss:
                            str = ss.readline()
                            print(str)
                            str = ss.readline()
                            sp = str.split('spans="')
                            spa = sp[1].split('"')
                            lenT = int(spa[0].split(':')[1])
                            aa = str.split('<v>')
                            aa = aa[1:]
                            for a in aa:
                                aaa = a.split('</v>')
                                sheet.append(aaa[0])
                            ss.close()
                        with open(op2n, "a") as newText:
                            print(sheetC)
                            title = sheetName[sheetC]
                            newText.write(title + "\n")
                            print(title)
                            print(lenT)
                            i = 0
                            text = ''
                            current = 0
                            while i < len(sheet):
                                if i % lenT == 0:
                                    if float(sheet[i]) == current:
                                        text += sharedString[current] + " " + "|" + " "
                                        current += 1
                                    else:
                                        text += sheet[i] + " " + "|" + " "
                                    newText.write(text + "\n")
                                    text = ''
                                else:
                                    if float(sheet[i]) == current:
                                        text += sharedString[current] + " " + "|" + " "
                                        current += 1
                                    else:
                                        text += sheet[i] + " " + "|" + " "
                                i += 1
                            newText.close()
                        sheetC += 1

                    # 内嵌文件
                    if na.endswith(".bin"):
                        op4 = outPath + "//" + fi + "//" + "office"
                        if not os.path.exists(op4):
                            os.makedirs(op4)
                        oleExtract(na, op4)
                    # 对应格式的内嵌文件
                    if na.endswith(".doc") or na.endswith(".docx") or na.endswith(".ppt") or na.endswith(
                            ".pptx") or na.endswith(".xls") or na.endswith(".xlsx"):
                        op3 = outPath + "//" + fi + "//" + "office"
                        if not os.path.exists(op3):
                            os.makedirs(op3)
                        op3n = op3 + "//" + name
                        shutil.copyfile(na, op3n)


# 解压pptx文件（实际上就是zip）
def deZipPPT(path_original):
    for file in os.listdir(path_original):
        if file.endswith('.pptx'):
            in_file = os.path.abspath(path_original + "//" + file)

            f = zipfile.ZipFile(in_file, "r")
            print(f.namelist())
            out_file = path_original + "//" + "in" + "//" + file.split(".")[0]
            if not os.path.exists(out_file):
                os.makedirs(out_file)
            # os.makedirs(out_file)
            for name in f.namelist():
                print(name)
                a = name.split("/")
                if len(a) > 1:
                    if a[0] == "ppt" and (a[1] == "embeddings" or a[1] == "media"):
                        f.extract(name, out_file)
                    if a[0] == "ppt" and (a[1] == "slides" and a[2] != "_rels"):
                        f.extract(name, out_file)


# 从解压的对应文件中提取出需要的信息
def getAllFilesPPT(path_o):
    inpath = os.path.abspath(path_o + "//" + "in")
    if not os.path.exists(inpath):
        os.makedirs(inpath)
    outPath = os.path.abspath(path_o + "//" + "out")
    if not os.path.exists(outPath):
        os.makedirs(outPath)
    file = os.listdir(inpath)
    for fi in file:
        fp = os.path.join(inpath, fi)
        if os.path.isdir(fp):
            print(fi)
            slide = 0
            for root, dirs, files in os.walk(fp):
                for name in files:
                    na = os.path.join(root, name)
                    print(na)
                    print(name)
                    # 图片
                    if na.endswith(".emf") or imghdr.what(na) != None:
                        op1 = outPath + "//" + fi + "//" + "image"
                        if not os.path.exists(op1):
                            os.makedirs(op1)
                        op1n = op1 + "//" + name
                        shutil.copyfile(na, op1n)
                    # 处理ppt内容
                    # 文本
                    if na.endswith(".xml"):
                        slide += 1
                        text = ""
                        with open(na, "r", encoding="utf-8") as bf:
                            stri = bf.readline()
                            stri = bf.readline()
                            aa = stri.split("<a:t>")
                            aa = aa[1:]
                            for a in aa:
                                aaa = a.split("</a:t>")
                                text += aaa[0]
                            bf.close()
                        op2 = outPath + "//" + fi + "//" + "text"
                        if not os.path.exists(op2):
                            os.makedirs(op2)
                        op2n = op2 + "//" + fi + ".txt"
                        with open(op2n, "a") as newText:
                            newText.write("第" + str(slide) + "页" + "\n")
                            try:
                                newText.write(text + "\n")
                            except:
                                UnicodeEncodeError
                            newText.close()

                    # 内嵌文件
                    if na.endswith(".bin"):
                        op4 = outPath + "//" + fi + "//" + "office"
                        if not os.path.exists(op4):
                            os.makedirs(op4)
                        oleExtract(na, op4)
                    # 对应格式的内嵌文件
                    if na.endswith(".doc") or na.endswith(".docx") or na.endswith(".ppt") or na.endswith(
                            ".pptx") or na.endswith(".xls") or na.endswith(".xlsx"):
                        op3 = outPath + "//" + fi + "//" + "office"
                        if not os.path.exists(op3):
                            os.makedirs(op3)
                        op3n = op3 + "//" + name
                        shutil.copyfile(na, op3n)


class NoOLEObjectStreamException(Exception):
    pass


# 字典对象，方便构建使用
class dictionary:
    def __init__(self, data):
        self.data = data
        self.dir = struct.unpack("<64sh2b3L16sL2Q3L", data)
        self.entryName = self.dir[0]  # 入口名称 64
        self.entryNameLength = self.dir[1]  # 入口名称字符串长度 2
        self.name = self.dir[0][0:self.dir[1]]
        self.entryType = self.dir[2]  # 1 入口类型 00空 03锁定的 01用户存储 04优先 05根存储 02用户流
        self.BRTreeColor = self.dir[3]  # 1 红黑树节点颜色 00红 01黑
        self.leftDid = self.dir[4]  # 4 左节点did 无则为-1
        self.rightDid = self.dir[5]  # 4 右节点did
        self.rootDid = self.dir[6]  # 4 根节点did
        self.sigID = self.dir[7]  # 16 标识符
        self.userFlag = self.dir[8]  # 4 用户标记
        self.createTime = self.dir[9]  # 8 创建时间
        self.lastWriteTime = self.dir[10]  # 8 最后修改时间
        self.sidStart = self.dir[11]  # 第一个存放的sid
        self.streamSize = self.dir[12]  # 流大小


# ole格式复合文件解析
def oleExtract(file, outPath):
    # ole复合文档头标识
    OLE_SIGNATURE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
    OLESIGN = b'\x01\x00O\x00l\x00e\x001\x000\x00N\x00a\x00t\x00i\x00v\x00e\x00\x00\x00'
    OLESIGN2 = b'\x02\x00O\x00l\x00e\x00P\x00r\x00e\x00s\x000\x000\x000\x00\x00\x00'
    # fileSize=os.path.getize(file)
    with open(file, "rb") as f:
        sig = f.read(8)  # 标识位 8
        print(sig)
        f.seek(0)
        head = f.read(512)
        if not sig == OLE_SIGNATURE:
            print("不是正确的OLE文件")
            return
        data = struct.unpack("<8s16sHHHHHHLLLLLLLLLL109L", head)
        sig16 = data[1]  # 16位标识
        cVersion = data[2]  # 2 文件格式修订号
        version = data[3]  # 2 文件格式版本号
        byteOrder = data[4]  # 寻址方式 FEFF小端 FFFE大端 2
        sectorSize = data[5]  # 2 扇区大小
        sectorSize = 2 ** sectorSize
        sSectorSize = data[6]  # 2 短扇区大小
        sSectorSize = 2 ** sSectorSize
        # 扇区总数
        # sectorNum=(fileSize-512)/sectorSize
        # 未使用10
        numStoreSAT = data[10]  # 4 存放扇区配置表的扇区数量
        print(numStoreSAT)
        firstDirStart = data[11]  # 4 存放目录的第一个扇区ID
        print(firstDirStart)
        # 未使用 4
        minStreamSize = data[13]  # 4 标准流的最小大小
        print(minStreamSize)
        firstSSecStart = data[14]  # 4 存放短扇区配置表的第一个扇区ID
        print(firstSSecStart)
        numStoreMSAT = data[15]  # 4 存放短扇区配置表的扇区数
        firstMainSATStart = data[16]  # 4 存放主扇区配置表的第一个扇区ID
        numStoreMainSAT = data[17]  # 存放主扇区配置表的扇区数
        print(numStoreMSAT)
        SATinHEAD = data[18:]  # HEADER中存放扇区配置表的首部分

        # 加载扇区配置表首部分 前436字节
        mainSAT = []
        for fat_sect in SATinHEAD:
            if fat_sect != 4294967295:  # 空闲位为-1
                mainSAT.append(fat_sect)
        print(mainSAT)

        # 加载主扇区配置表
        sector = firstMainSATStart  # did开始的sid
        print(sector)
        while sector != 4294967294 and sector != 4294967295:  # -2为最后一个 为-1是freesid
            # 读扇区
            f.seek(0)
            f.seek(512 + sector * sectorSize)
            dataMS = f.read(sectorSize)
            print(dataMS)
            # 每4位一个sid
            dif_values = [x for x in struct.unpack('<{0}L'.format(sectorSize / 4), dataMS)]
            # 最后一位是指向下一个did的sid
            next = dif_values.pop()
            for value in dif_values:
                if value != 4294967295:  # freesid表示空闲 不为空则加载
                    mainSAT.append(value)
            sector = next

        # 加载扇区配置表
        SAT = []
        for fat_sect in mainSAT:
            print(fat_sect)
            # 主扇区配置表（MSAT：master sector allocation table）是一个SID数组，指明了所有用于存放扇区配置表（SAT：sector allocation table）的sector的SID
            f.seek(0)
            f.seek(512 + fat_sect * sectorSize)
            dataS = f.read(sectorSize)
            # 若读取的数据小于一个扇区大小
            if len(dataS) != sectorSize:
                print('broken FAT (invalid sector size {0} != {1})'.format(len(data), sectorSize))
            else:
                for value in struct.unpack('<{0}L'.format(int(sectorSize / 4)), dataS):
                    SAT.append(value)
        SAT2 = []
        # 去除等于-1的freeid
        for sa in SAT:
            if sa != 4294967295:
                SAT2.append(sa)

        print(SAT2)

        # 获取字典
        fdid = firstDirStart
        dict = []
        print("获取字典")
        while fdid != 4294967294:
            did = 512 + fdid * sectorSize
            # 一个512的扇区可以放4个目录入口
            for i in range(int(sectorSize / 128)):
                f.seek(0)
                f.seek(did)
                directory = f.read(128)
                dict.append(dictionary(directory))
                did += 128
            fdid = SAT2[fdid]  # 从扇区配置表中找是否有下一个目录存在
        # FID为文件流所在扇区id FL为文件流长度
        FID = 0
        FL = 0
        for dir in dict:
            print(dir.name)
            print(dir.streamSize)
            print(dir.sidStart)
            if dir.name == OLESIGN:
                FID = dir.sidStart
                FL = dir.streamSize
                print(FL, FID)
        try:
            if FL == 0:
                raise NoOLEObjectStreamException
        except:
            NoOLEObjectStreamException
            print('未在目录中匹配到OLE10NATIVE')
            return
        # 不为短流 直接找到位置
        if FL > minStreamSize:
            pos = 512 + FID * sectorSize
            f.seek(0)
            f.seek(pos)
            stream = f.read(FL)
        elif FL > 0:
            # 加载短扇区配置表
            print("开始加载短扇区配置表")
            mSAT = []
            start = firstSSecStart
            print(start)
            f.seek(0)
            f.seek(512 + start * sectorSize)
            dataMinSAT = f.read(sectorSize)
            print(SAT2[start])
            print(dataMinSAT)
            for value in struct.unpack('<{0}L'.format(int(sectorSize / 4)), dataMinSAT):
                mSAT.append(value)
            mSAT2 = []
            for msa in mSAT:
                if msa != 4294967295:
                    mSAT2.append(msa)
            print(mSAT2)
            pos = 512 + FID * sectorSize + FID * sSectorSize
            f.seek(0)
            f.seek(pos)
            stream = f.read(FL)
        # print(stream)
        # 前四字节为整个流的大小
        l = struct.unpack("<I", stream[0:4])
        print(l)
        # 后2字节无意义 跳过
        stream = stream[6:]
        # 文件名称 以0结尾
        i = 0
        while i < len(stream):
            if stream[i] == 0:
                break
            i += 1
        name1 = stream[0:i].decode("gbk", "ignore")
        print(name1)
        stream = stream[i + 1:]
        # 文件源路径 以0结尾
        i = 0
        while i < len(stream):
            if stream[i] == 0:
                break
            i += 1
        name2 = stream[0:i].decode("gbk", "ignore")
        print(name2)
        # 后面有8位未知 跳过
        stream = stream[i + 9:]
        # 目标路径 以0结尾
        i = 0
        while i < len(stream):
            if stream[i] == 0:
                break
            i += 1
        name3 = stream[0:i].decode("gbk", "ignore")
        print(name3)
        stream = stream[i + 1:]
        # 原始数据长度
        oFLength = struct.unpack("<I", stream[0:4])[0]
        stream = stream[4:oFLength + 4]
        outP = outPath + "//" + name1
        try:
            with open(outP, "wb") as out:
                out.write(stream)
                out.close()
            print("成功提取：" + name1)
        except:
            OSError
            print("路径名非法")


def main():
    # 文件路径
    root = tk.Tk()
    root.withdraw()
    path_original = filedialog.askdirectory()
    print("请选择解析的文件夹路径")
    print(path_original)
    value = input("请选择word,excel,ppt,pdf或cebx:\n")
    if value == 'word':
        checkIfDocx(path_original)
        deZipWord(path_original)
        getAllFilesWord(path_original)
    elif value == 'excel':
        checkIfXlsx(path_original)
        deZipExcel(path_original)
        getAllFilesExcel(path_original)
    elif value == 'ppt':
        checkIfPPTX(path_original)
        deZipPPT(path_original)
        getAllFilesPPT(path_original)
    elif value== 'pdf':
        extractPDF(path_original)


if __name__ == '__main__':
    main()
