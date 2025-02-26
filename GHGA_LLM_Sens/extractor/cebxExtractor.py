import imghdr
import os
import struct
import zlib
import numpy as np


# SXC格式解压缩
def DecompressSXC(way1, BSLen, current):
    print("开始执行SXC文件解压函数")
    with open(way1, "rb") as ff:
        ff.seek(25)
        method = ff.read(26)
        print(method)
        MET = struct.unpack("<2B6I", method)
        MET = list(MET)
        print(MET)
        if MET[1] == 3:
            print(MET[5])
            DecompressStructData(MET, current, ff)
            DecompressDataContainerMap(MET, current, ff)
        else:
            print("标识位为：" + str(MET[1]))
            print("存在未考虑情况，请重新分析标识位")
        ff.close()

# 解压缩sxc格式中的结构信息部分
def DecompressStructData(MET, current, ff):
    ff.seek(MET[2])
    datastruct = ff.read(MET[3])
    print(MET[3])
    # format4 = "<" + str(MET[3]) + "B"
    # ds = struct.unpack(format4, datastruct)
    # pathDS = "C:\\Users\\byz\\Desktop\\cebx\\sxc\\DataStruct\\" + "DataStructCompressed_" + str(current) + ".raw"
    # # 写入
    # with open(pathDS, "wb") as fs:
    #     inStream = struct.pack(format4, *ds)
    #     fs.write(inStream)
    #     print("写入raw文件完成")
    #     fs.close()
    # # 解压缩
    # zlibFile = open(pathDS, "rb")
    address2 = "DataStruct_" + str(current) + ".raw"
    path2 = "C:\\Users\\byz\\Desktop\\cebx\\sxc\\DataStruct\\" + address2
    endFile = open(path2, "wb")
    decompressobj = zlib.decompressobj()
    # data = zlibFile.read(MET[3])
    endFile.write(decompressobj.decompress(datastruct))
    endFile.write(decompressobj.flush())
    endFile.close()
    print("解压缩DataStruct完成")

# 解压缩sxc格式中的数据容器部分
def DecompressDataContainer(MET, current, ff, ucd, ddc, count):
    # ff.seek(MET[4])
    # datastruct = ff.read(MET[5])
    ff.seek(MET[4] + ddc)
    datastruct = ff.read(ucd)
    print(MET[5])
    print(ddc)
    # format4 = "<" + str(MET[5]) + "B"
    # ds = struct.unpack(format4, datastruct)
    # pathDS = "C:\\Users\\byz\\Desktop\\cebx\\sxc\\DataContainer\\" + "DataContainerCompressed_" + str(current) + ".raw"
    # # 由于要多次解压，加入判断减少冗余
    # if not os.path.exists(pathDS):
    #     with open(pathDS, "wb") as fs:
    #         inStream = struct.pack(format4, *ds)
    #         fs.write(inStream)
    #         print("写入raw文件完成")
    #         fs.close()
    # # 解压缩
    # zlibFile = open(pathDS, "rb")
    address2 = "DataContainer_" + str(current) + "_" + str(count) + ".raw"
    path2 = "C:\\Users\\byz\\Desktop\\cebx\\sxc\\DataContainer\\" + address2
    endFile = open(path2, "wb")
    decompressobj = zlib.decompressobj()
    # zlibFile.seek(ddc)
    # data = zlibFile.read(ucd)
    endFile.write(decompressobj.decompress(datastruct))
    endFile.write(decompressobj.flush())
    endFile.close()
    print("解压缩DataContainer完成")
    ddc = ddc + ucd
    return ddc

# 根据数据容器map信息解压数据容器
def DecompressDataContainerMap(MET, current, ff):
    ff.seek(MET[6])
    datastruct = ff.read(2)
    DataMap = struct.unpack("<2B", datastruct)
    DataMap = list(DataMap)
    print(DataMap)
    if DataMap[0] == 0 and DataMap[1] == 1:
        ff.seek(MET[6] + 2)
        datastruct = ff.read(MET[7] - 2)
        # format4 = "<" + str(MET[7] - 2) + "B"
        # ds = struct.unpack(format4, datastruct)
        # pathDS = "C:\\Users\\byz\\Desktop\\cebx\\sxc\\DataContainerMap\\" + "DataContainerMapCompressed_" + str(
        #     current) + ".raw"
        # # 写入
        # with open(pathDS, "wb") as fs:
        #     inStream = struct.pack(format4, *ds)
        #     fs.write(inStream)
        #     print("写入raw文件完成")
        #     fs.close()
        # # 解压缩
        # zlibFile = open(pathDS, "rb")
        address2 = "DataContainerMap_" + str(current) + ".raw"
        path2 = "C:\\Users\\byz\\Desktop\\cebx\\sxc\\DataContainerMap\\" + address2
        endFile = open(path2, "wb")
        decompressobj = zlib.decompressobj()
        # data = zlibFile.read(MET[7] - 2)
        endFile.write(decompressobj.decompress(datastruct))
        endFile.write(decompressobj.flush())
        endFile.close()
        with open(path2, "rb") as dm:
            cc = dm.read(8)
            print("未合并容器条目数 合并容器条目数")
            ccc = struct.unpack("<2I", cc)
            print(ccc)
            cur = 8
            ddc = 0
            uclis = []
            if ccc[0] != 0:
                for i in range(ccc[0]):
                    dm.seek(cur + i * 8)
                    uc = dm.read(8)
                    ucd = struct.unpack("<2I", uc)
                    ucd = list(ucd)
                    uclis.append(ucd[0])
                    ddc = DecompressDataContainer(MET, current, ff, ucd[1], ddc, i)
            cur += 8 * ccc[0]
            searchNodes(uclis, current)
            for m in range(ccc[1]):
                clis = []
                llis = []
                dm.seek(cur)
                ca = dm.read(8)
                cas = struct.unpack("<2I", ca)
                print(cas)
                ddc = DecompressDataContainer(MET, current, ff, cas[1], ddc, m + ccc[0])
                dm.seek(cur + 8)
                cdi = dm.read(8 * cas[0])
                form = "<" + str(cas[0] * 2) + "I"
                cdim = struct.unpack(form, cdi)
                cidm = list(cdim)
                for a in range(cas[0]):
                    clis.append(cidm[a * 2])
                    llis.append(cidm[a * 2 + 1])
                cur += 8 + 8 * cas[0]
                searchNodesCombined(clis, current, llis)
            dm.close()
        print("解压缩DataContainerMap完成")

    else:
        print("存在编码方式，仍需分析")


# bitStream读写函数 进行迭代
def BitStreamRW(current, f, bp):
    f.seek(current)
    bsign = f.read(3 + bp)
    current += 3 + bp
    # print(bsign)
    forma3 = "<B" + transformBP(bp) + "2B"
    BSIGN = struct.unpack("<BI2B", bsign)
    BSIGN = list(BSIGN)
    print(BSIGN)
    BSLen = BSIGN[1]
    # bs流压缩方法校验 02为zlib 01为保留格式（即sxc
    if BSIGN[2] != 2 & BSIGN[3] != 255:
        print("一眼CEBX，鉴定为SXC")
        f.seek(current)
        bstream = f.read(BSLen)
        format = "<" + str(BSLen) + "B"
        BSTREAM = struct.unpack(format, bstream)
        address1 = "Infile_NotZlib" + str(current) + ".raw"
        way1 = "C:\\Users\\byz\\Desktop\\cebx\\sxc\\" + address1
        # 写入
        with open(way1, "wb") as ff:
            inStream = struct.pack(format, *BSTREAM)
            ff.write(inStream)
            print("写入raw文件完成")
            ff.close()
        DecompressSXC(way1, BSLen, current)
        return current + BSLen
    # ZLIB格式压缩直接解压
    f.seek(current)
    bstream = f.read(BSLen)
    # format = "<" + str(BSLen) + "B"
    # BSTREAM = struct.unpack(format, bstream)
    # address1 = "Infile_" + str(current) + ".raw"
    # way1 = "C:\\Users\\byz\\Desktop\\cebx\\Infile\\" + address1
    # # 写入
    # with open(way1, "wb") as ff:
    #     inStream = struct.pack(format, *BSTREAM)
    #     ff.write(inStream)
    #     print("写入raw文件完成")
    #     ff.close()
    # 解压缩
    # zlibFile = open(way1, "rb")
    address2 = "OutFile_" + str(current) + ".raw"
    way2 = "C:\\Users\\byz\\Desktop\\cebx\\OutFile\\" + address2
    endFile = open(way2, "wb")
    decompressobj = zlib.decompressobj()
    # data = zlibFile.read(BSLen)
    endFile.write(decompressobj.decompress(bstream))
    endFile.write(decompressobj.flush())
    print("解压缩完成")
    endFile.close()
    current += BSLen
    return current


# 标记文件类型
def labelFile():
    path = "C:\\Users\\byz\\Desktop\\cebx\\OutFile\\"
    list_file = os.listdir(path)
    for fil in list_file:
        ext = os.path.splitext(fil)  # 返回文件名和后缀
        newfile = ext[0]
        print(fil)
        # 调用imghdr方法，判断文件类型
        imageType = imghdr.what(path + fil)
        if imageType != None:
            newfile = newfile + "." + imageType
        else:
            with open(path + fil, mode="rb") as file:
                label = file.read(2)
                print(label)
                HEAD = struct.unpack("<2B", label)
                HEAD = list(HEAD)
                print(HEAD)
                file.close()
            if HEAD[0] == 33 and HEAD[1] == 69:
                newfile = newfile + ".EFC"
            elif HEAD[0] == 33 and HEAD[1] == 73:
                newfile = newfile + ".IFC"
            elif HEAD[0] == 64 and HEAD[1] == 67:
                newfile = newfile + ".bsg"
            else:
                newfile = newfile + ".xml"

        os.rename(os.path.join(path, fil), os.path.join(path, newfile))


# 找到数据容器中对应的节点信息
def searchNodesCombined(list, current, llis):
    typeList = []
    print(list)
    print(llis)
    i = 0
    for node in list:
        with open("C:\\Users\\byz\\Desktop\\cebx\\sxc\\schemaNode.txt", "r") as txt:
            snl = txt.readlines()
            for sn in snl:
                ss = sn[:-1].split(" ")
                if str(node) == ss[0]:
                    if len(ss) < 3:
                        typeList.append(ss[0] + " " + ss[1] + " " + str(llis[i]))
                    else:
                        typeList.append(ss[0] + " " + ss[3] + " " + str(llis[i]))
        i += 1
    with open("C:\\Users\\byz\\Desktop\\cebx\\sxc\\dataMap.txt", "a") as txt:
        # 节点ID 节点类型
        txt.write("\n" + str(current) + " " + str(typeList))
        # for tl in typeList:
        #     print(tl)
        #     txt.write(" "+str(tl[0]) + " " + str(tl[1])+" "+str(tl[2]))
        txt.close()


# 找到数据容器中对应的节点信息
def searchNodes(list, current):
    typeList = []
    print(list)
    for node in list:
        with open("C:\\Users\\byz\\Desktop\\cebx\\sxc\\schemaNode.txt", "r") as txt:
            snl = txt.readlines()
            for sn in snl:
                ss = sn[:-1].split(" ")
                if str(node) == ss[0]:
                    if len(ss) < 3:
                        typeList.append(ss[0] + " " + ss[1])
                    else:
                        typeList.append(ss[0] + " " + ss[3])
    with open("C:\\Users\\byz\\Desktop\\cebx\\sxc\\dataMap.txt", "a") as txt:
        # 节点ID 节点类型
        txt.write("\n" + str(current) + " " + str(typeList))
        # for tl in typeList:
        #     txt.write(" "+str(tl[0])+" "+str(tl[1]))
        txt.close()


# 写xml文件
def writeXMLString(dms, x, file, a, count):
    file.seek(x)
    length = int(dms[(a + 1) * 3])
    f1 = file.read(length)
    # print(f1)
    form = "<" + str(length) + "c"
    f1f = struct.unpack(form, f1)
    fil = "C:\\Users\\byz\\Desktop\\cebx\\file\\XML_v1_" + str(count) + ".txt"
    with open(fil, "a", encoding="utf-8") as out:
        string = getNodeName(dms[3 * a + 1])
        itm = "<" + string + ">"
        itm2 = "</" + string + ">"
        out.write(itm + f1.decode("utf-8", "ignore") + itm2 + "\n")
        out.close
    # with open(fil, "ab") as out:
    #     string=getNodeName(dms[3*a+1])
    #     itm = "<"+string+">"
    #     itm2 = "</"+string+">"
    #     n = "\n"
    #     out.write(itm.encode(encoding="utf-8"))
    #     inStream = struct.pack(form, *f1f)
    #     out.write(inStream)
    #     out.write(itm2.encode(encoding="utf-8"))
    #     out.write(n.encode(encoding="utf-8"))
    #     out.close()
    return x + length


# 写xml文件
def writeXMLInt(dms, x, file, a, count):
    file.seek(x)
    length = int(dms[(a + 1) * 3])
    f1 = file.read(length)
    # print(f1.decode(encoding="utf-8"))
    form = "<" + str(int(length / 4)) + "I"
    f1f = struct.unpack(form, f1)
    # print(f1f)
    fil = "C:\\Users\\byz\\Desktop\\cebx\\file\\XML_v1_" + str(count) + ".txt"
    with open(fil, "a", encoding="utf-8") as out:
        string = getNodeName(dms[3 * a + 1])
        itm = "<" + string + ">"
        itm2 = "</" + string + ">"
        out.write(itm + str(f1f) + itm2 + "\n")
        out.close()
    return x + length


# 写xml文件
def writeXMLFloat(dms, x, file, a, count):
    print("float")
    file.seek(x)
    length = int(dms[(a + 1) * 3])
    f1 = file.read(length)
    # print(f1.decode(encoding="utf-8"))
    form = "<" + str(int(length / 8)) + "d"
    f1f = struct.unpack(form, f1)
    # print(f1f)
    fil = "C:\\Users\\byz\\Desktop\\cebx\\file\\XML_v1_" + str(count) + ".txt"
    with open(fil, "a", encoding="utf-8") as out:
        string = getNodeName(dms[3 * a + 1])
        itm = "<" + string + ">"
        itm2 = "</" + string + ">"
        out.write(itm + str(f1f) + itm2 + "\n")
        out.close()
    return x + length


# 写xml文件
def writeXMLBool(dms, x, file, a, count):
    print("bool")
    file.seek(x)
    length = int(dms[(a + 1) * 3])
    f1 = file.read(length)
    # print(f1.decode(encoding="utf-8"))
    form = "<" + str(int(length)) + "B"
    f1f = struct.unpack(form, f1)
    # print(f1f)
    fil = "C:\\Users\\byz\\Desktop\\cebx\\file\\XML_v1_" + str(count) + ".txt"
    with open(fil, "a", encoding="utf-8") as out:
        string = getNodeName(dms[3 * a + 1])
        itm = "<" + string + ">"
        itm2 = "</" + string + ">"
        out.write(itm + str(f1f) + itm2 + "\n")
        out.close()
    return x + length


# 写xml文件 未合并
def writeXMLStringNC(dms, file, a, count, length):
    print(length)
    f1 = file.read(length)
    # print(f1)
    form = "<" + str(length) + "c"
    fil = "C:\\Users\\byz\\Desktop\\cebx\\file\\XML_v1_" + str(count) + ".txt"
    with open(fil, "a", encoding="utf-8") as out:
        string = getNodeName(dms[2 * a + 1])
        itm = "<" + string + ">"
        itm2 = "</" + string + ">"
        out.write(itm + f1.decode("utf-8", "ignore") + itm2 + "\n")
        out.close


# 写xml文件
def writeXMLIntNC(dms, file, a, count, length):
    print(length)
    f1 = file.read(length)
    print(f1)
    form = "<" + str(int(length / 4)) + "I"
    f1f = struct.unpack(form, f1)
    # print(f1f)
    fil = "C:\\Users\\byz\\Desktop\\cebx\\file\\XML_v1_" + str(count) + ".txt"
    with open(fil, "a", encoding="utf-8") as out:
        string = getNodeName(dms[2 * a + 1])
        itm = "<" + string + ">"
        itm2 = "</" + string + ">"
        out.write(itm + str(f1f) + itm2 + "\n")
        out.close()


# 写xml文件
def writeXMLFloatNC(dms, file, a, count, length):
    print("float")
    f1 = file.read(length)
    # print(f1.decode(encoding="utf-8"))
    form = "<" + str(int(length / 8)) + "d"
    f1f = struct.unpack(form, f1)
    # print(f1f)
    fil = "C:\\Users\\byz\\Desktop\\cebx\\file\\XML_v1_" + str(count) + ".txt"
    with open(fil, "a", encoding="utf-8") as out:
        string = getNodeName(dms[2 * a + 1])
        itm = "<" + string + ">"
        itm2 = "</" + string + ">"
        out.write(itm + str(f1f) + itm2 + "\n")
        out.close()


# 写xml文件
def writeXMLBoolNC(dms, file, a, count, length):
    print("bool")
    f1 = file.read(length)
    # print(f1.decode(encoding="utf-8"))
    form = "<" + str(int(length)) + "B"
    f1f = struct.unpack(form, f1)
    # print(f1f)
    fil = "C:\\Users\\byz\\Desktop\\cebx\\file\\XML_v1_" + str(count) + ".txt"
    with open(fil, "a", encoding="utf-8") as out:
        string = getNodeName(dms[2 * a + 1])
        itm = "<" + string + ">"
        itm2 = "</" + string + ">"
        out.write(itm + str(f1f) + itm2 + "\n")
        out.close()


# 获取对应ID的结点名称
def getNodeName(ID):
    with open("C:\\Users\\byz\\Desktop\\cebx\\sxc\\schemaNode.txt", "r") as txt:
        snl = txt.readlines()
        for sn in snl:
            ss = sn[:-1].split(" ")
            if ID == ss[0]:
                return ss[2]


# dataMap文件格式转换 还原xml文件
def dataMapAnaylize():
    with open("C:\\Users\\byz\\Desktop\\cebx\\sxc\\dataMap.txt", "r") as txt:
        dml = txt.readlines()
        id = ""
        size = 0
        for dm in dml:
            string = str(dm[:-1])
            string = string.replace('[', '')
            string = string.replace(']', '')
            string = string.replace("'", '')
            string = string.replace(',', '')
            dms = string.split(" ")
            # print(dms)
            if id == dms[0]:
                form = "C:\\Users\\byz\\Desktop\\cebx\\sxc\\DataContainer\\" + "DataContainer_" + dms[
                    0] + "_" + str(size) + ".raw"
                with open(form, "rb") as file:
                    x = 0
                    for a in range(int(len(dms) / 3)):
                        if dms[3 * a + 2] == "String":
                            x = writeXMLString(dms, x, file, a, id)
                        elif dms[3 * a + 2] == "Integer":
                            x = writeXMLInt(dms, x, file, a, id)
                        elif dms[3 * a + 2] == "Float":
                            x = writeXMLFloat(dms, x, file, a, id)
                        elif dms[3 * a + 2] == "Bool":
                            x = writeXMLBool(dms, x, file, a, id)
                size += 1
            else:
                size = 0
                id = dms[0]
                # 未合并的datacontainer
                if len(dms) > 2:
                    for a in range(int(len(dms) / 2)):
                        form = "C:\\Users\\byz\\Desktop\\cebx\\sxc\\DataContainer\\" + "DataContainer_" + dms[
                            0] + "_" + str(a) + ".raw"
                        length = os.path.getsize(form)
                        with open(form, "rb") as file:
                            if dms[2 * a + 2] == "String":
                                writeXMLStringNC(dms, file, a, id, length)
                            elif dms[2 * a + 2] == "Integer":
                                writeXMLIntNC(dms, file, a, id, length)
                            elif dms[2 * a + 2] == "Float":
                                writeXMLFloatNC(dms, file, a, id, length)
                            elif dms[2 * a + 2] == "Bool":
                                writeXMLBoolNC(dms, file, a, id, length)
                    size += 1


# 判断属性或元素结点的数据类型
def whichDataType(n):
    if n == 0:
        return "Bool"
    elif n == 1:
        return "Integer"
    elif n == 3:
        return "Float"
    else:
        return "String"


# BSG文件 读取节点信息
def readSchemaNodes(c, bsg, i):
    bsg.seek(c)
    tpye = bsg.read(1)
    c += 1
    type = struct.unpack("<B", tpye)
    type = list(type)
    if type[0] == 0:
        print("unknown")
        bsg.seek(c)
        unko = bsg.read(12)
        c += 12
        unk = struct.unpack("<3I", unko)
        unk = list(unk)
        print(unk)
        bsg.seek(c)
        chi = bsg.read(unk[2] * 4)
        form2 = "<" + str(unk[2]) + "I"
        chil = struct.unpack(form2, chi)
        chil = list(chil)
        print(chil)
        c += unk[2] * 4
        with open("C:\\Users\\byz\\Desktop\\cebx\\sxc\\schemaNode.txt", "a") as txt:
            # 节点ID 节点类型
            txt.write("\n" + str(i))
            txt.write(" " + "unknown")
            txt.close()
    elif type[0] == 1:
        print("属性节点")
        bsg.seek(c)
        nle = bsg.read(4)
        nl = struct.unpack("<I", nle)
        nl = list(nl)
        print(nl)
        c += 4
        bsg.seek(c)
        nna = bsg.read(nl[0] * 2)
        nnaa = list(nna)
        print(nna)
        c += nl[0] * 2
        bsg.seek(c)
        dty = bsg.read(2)
        dtype = struct.unpack("<2B", dty)
        dtype = list(dtype)
        type = whichDataType(dtype[0])
        print(type)
        c += 2
        print(dtype)
        with open("C:\\Users\\byz\\Desktop\\cebx\\sxc\\schemaNode.txt", "a") as txt:
            # 节点ID 节点类型 节点名称 数据类型
            txt.write("\n" + str(i))
            txt.write(" " + "attribute")
            nameStr = ""
            for m in range(nl[0]):
                nameStr = nameStr + chr(nnaa[2 * m])
            txt.write(" " + nameStr + " " + type)
            txt.close()
    elif type[0] == 2:
        print("元素节点")
        bsg.seek(c)
        nle = bsg.read(4)
        nl = struct.unpack("<I", nle)
        nl = list(nl)
        print(nl)
        c += 4
        bsg.seek(c)
        nna = bsg.read(nl[0] * 2)
        nnaa = list(nna)
        print(nna)
        print(nnaa)
        c += nl[0] * 2
        bsg.seek(c)
        dty = bsg.read(14)
        c += 14
        dtype = struct.unpack("<B2IBI", dty)
        dtype = list(dtype)
        print(dtype)
        type = whichDataType(dtype[0])
        print(type)
        att = []
        if dtype[4] != 0:
            bsg.seek(c)
            c += 4 * dtype[4]
            atts = bsg.read(4 * dtype[4])
            form = "<" + str(dtype[4]) + "I"
            att = struct.unpack(form, atts)
            att = list(att)
            print(att)
        bsg.seek(c)
        chco = bsg.read(4)
        c += 4
        chc = struct.unpack("<I", chco)
        chc = list(chc)
        print(chc)
        chi = bsg.read(chc[0] * 4)
        form2 = "<" + str(chc[0]) + "I"
        chil = struct.unpack(form2, chi)
        chil = list(chil)
        print(chil)
        c += chc[0] * 4
        with open("C:\\Users\\byz\\Desktop\\cebx\\sxc\\schemaNode.txt", "a") as txt:
            # 节点ID 节点类型 节点名称 数据类型 节点属性集合 节点子节点集合
            txt.write("\n" + str(i))
            txt.write(" " + "element")
            nameStr = ""
            for m in range(nl[0]):
                nameStr = nameStr + chr(nnaa[2 * m])
            txt.write(" " + nameStr + " " + type + " " + str(att) + " " + str(chil))
            txt.close()
    if type[0] == 3:
        print("sequence")
        bsg.seek(c)
        unko = bsg.read(12)
        c += 12
        unk = struct.unpack("<3I", unko)
        unk = list(unk)
        print(unk)
        bsg.seek(c)
        chi = bsg.read(unk[2] * 4)
        form2 = "<" + str(unk[2]) + "I"
        chil = struct.unpack(form2, chi)
        chil = list(chil)
        print(chil)
        c += unk[2] * 4
        with open("C:\\Users\\byz\\Desktop\\cebx\\sxc\\schemaNode.txt", "a") as txt:
            # 节点ID 节点类型
            txt.write("\n" + str(i))
            txt.write(" " + "sequence")
            txt.close()
    if type[0] == 4:
        print("choice")
        bsg.seek(c)
        unko = bsg.read(12)
        c += 12
        unk = struct.unpack("<3I", unko)
        unk = list(unk)
        print(unk)
        bsg.seek(c)
        chi = bsg.read(unk[2] * 4)
        form2 = "<" + str(unk[2]) + "I"
        chil = struct.unpack(form2, chi)
        chil = list(chil)
        print(chil)
        c += unk[2] * 4
        with open("C:\\Users\\byz\\Desktop\\cebx\\sxc\\schemaNode.txt", "a") as txt:
            # 节点ID 节点类型
            txt.write("\n" + str(i))
            txt.write(" " + "choice")
            txt.close()
    if type[0] == 5:
        print("all")
        bsg.seek(c)
        unko = bsg.read(12)
        c += 12
        unk = struct.unpack("<3I", unko)
        unk = list(unk)
        print(unk)
        bsg.seek(c)
        chi = bsg.read(unk[2] * 4)
        form2 = "<" + str(unk[2]) + "I"
        chil = struct.unpack(form2, chi)
        chil = list(chil)
        print(chil)
        c += unk[2] * 4
        with open("C:\\Users\\byz\\Desktop\\cebx\\sxc\\schemaNode.txt", "a") as txt:
            # 节点ID 节点类型
            txt.write("\n" + str(i))
            txt.write(" " + "all")
            txt.close()
    return c


# BSG文件解析
def BSGAnaylize():
    path = "C:\\Users\\byz\\Desktop\\cebx\\OutFile\\"
    list_file = os.listdir(path)
    for fil in list_file:
        ext = os.path.splitext(fil)  # 返回文件名和后缀
        if ".bsg" == ext[1]:
            with open(path+fil, "rb") as bsg:
                fh = bsg.read(22)
                print("版本信息")
                print(fh)
                bsg.seek(22)
                checkMD5 = bsg.read(16)
                check = struct.unpack("<16B", checkMD5)
                check = list(check)
                print("MD5校验码")
                print(check)
                bsg.seek(38)
                nn = bsg.read(4)
                NodeNum = struct.unpack("<I", nn)
                nodeN=list(NodeNum)
                nodeN=NodeNum[0]
                print("节点数量")
                print(NodeNum)
                bsg.seek(42)
                curr = 42
                i = 0
                while i < nodeN:
                    curr = readSchemaNodes(curr, bsg, i)
                    print("第")
                    print(i)
                    i += 1
                bsg.close()
                # snt = bsg.read(5)
                # print(snt)
                # SchemaNodeType = struct.unpack("<BI", snt)
                # print(SchemaNodeType)
                # bsg.seek(47)
                # name = bsg.read(70)
                # print(name)
                # Nam = struct.unpack("<17B2IB11I", name)
                # print(Nam)
                # bsg.seek(117)
                # name2 = bsg.read(25)
                # print(name2)
                # Nam2 = struct.unpack("<BI20B", name2)
                # print(Nam2)
                # bsg.close()



# 解压缩图像数据包（大量小体积文件）
def decompressIFC():
    path = "C:\\Users\\byz\\Desktop\\cebx\\OutFile\\"
    list_file = os.listdir(path)
    for fil in list_file:
        ext = os.path.splitext(fil)  # 返回文件名和后缀
        print(ext)
        if ".IFC" == ext[1]:
            size = os.path.getsize(path+fil)
            print(size)
            with open(path+fil, "rb") as ifc:
                print("文件类型 版本 压缩单位 压缩方法")
                header = ifc.read(10)
                print(header)
                Head = struct.unpack("<2I2B", header)
                Head = list(Head)
                print(Head)
                if Head[1] == 2:
                    print("v1.2")
                    return
                # 主索引入口位于文件末尾
                ifc.seek(size - 4)
                mie = ifc.read(4)
                mien = struct.unpack("<I", mie)
                mien = list(mien)
                mie = mien[0]
                print(mie)
                ifc.seek(mie)
                print("索引数量 段最大粒度 段索引位置 图像数量")
                mi = ifc.read(13)
                min = struct.unpack("<IHBIH", mi)
                min = list(min)
                print(min)
                # 找到段索引
                dsp = min[3]
                ifc.seek(dsp)
                ds = ifc.read(14)
                print("数据段位置 数据偏移 数据长度 图像宽度 图像高度")
                dsi = struct.unpack("<2I3H", ds)
                dsi = list(dsi)
                print(dsi)
                for i in range(min[0]):
                    decompressImageData(dsi[0], dsi[1], dsi[2], ifc, ext[0], i)
                ifc.close()


def decompressImageData(p, e, s, file, fn, i):
    # 解压缩
    print(p, e, s, i)
    address2 = fn + "_Image" + "_" + str(i) + ".raw"
    way2 = "C:\\Users\\byz\\Desktop\\cebx\\OutFile\\" + address2
    endFile = open(way2, "wb")
    decompressobj = zlib.decompressobj()
    file.seek(p + e + i * s)
    data = file.read(s)
    endFile.write(decompressobj.decompress(data))
    endFile.write(decompressobj.flush())
    print("解压缩完成")
    endFile.close()
    size = os.path.getsize(way2)
    # 给像素数据添加文件头 这个部分问题暂时无法解决，文档中给的条件太少了，缺少图像文件中的调色板部分数据
    # headl = []
    # with open(way2, "rb") as shuju:
    #     sj = shuju.read(size)
    #     formate = "<" + str(size) + "B"
    #     sjj = struct.unpack(formate, sj)
    #     headl = list(sjj)
    #     shuju.close()
    # path="C:\\Users\\byz\\Desktop\\cebx\\OutFile\\OutFile_1756281.raw"
    # with open(path,"rb") as shu2:
    #     shu2.seek(54)
    #     shu=shu2.read(1024)
    #     sh=struct.unpack("<256I",shu)
    #     head3=list(sh)
    # Head = [19778, size + 1078, 0, 0, 1078]
    # head2 = [40, 212, 24, 1, 8, 0, 0, 0,0,256, 256]
    # Head = np.append(Head, head2)
    # Head=np.append(Head,head3)
    # print(Head)
    # Head = np.append(Head, headl)
    # address3 = "OutFileBMP_3_" + str(current) + ".raw"
    # format2 = "<HI2H4I2H262I" + str(size) + "B"
    # way3 = fn+"_ImageR"+"_"+str(i) + ".raw"
    # with open(way3, "wb") as f:
    #     a = struct.pack(format2, *Head)
    #     f.write(a)
    #     f.close()


# 解压字体数据包
def decompressEFC():
    path = "C:\\Users\\byz\\Desktop\\cebx\\OutFile\\"
    list_file = os.listdir(path)
    for fil in list_file:
        ext = os.path.splitext(fil)  # 返回文件名和后缀
        if ".EFC" == ext[1]:
            with open(path+fil, "rb") as efc:
                print("文件类型4 版本4 压缩单位1 压缩方法1 索引入口4")
                header = efc.read(14)
                print(header)
                Head = struct.unpack("<2I2BI", header)
                Head = list(Head)
                print(Head)
                # efc.seek(520953)
                # ind = efc.read(28)
                # index = struct.unpack("<7I", ind)
                # index = list(index)
                # print(index)


# 根据bp偏移判断解包的长度
def transformBP(bp):
    if bp == 1:
        return "B"
    elif bp == 2:
        return "H"
    elif bp == 4:
        return "I"
    elif bp == 8:
        return "Q"


# 解压nametable文件
def decompressNameTable(f,current,length):
    f.seek(current)
    endFile = open(r"C:\Users\byz\Desktop\cebx\NameTable.txt", "wb")
    decompressobj = zlib.decompressobj()
    data = f.read(length)
    endFile.write(decompressobj.decompress(data))
    endFile.write(decompressobj.flush())
    print("解压缩NameTable完成")

# 解压itemlist文件
def decompressItemList(f,current,length):
    f.seek(current)
    endFile = open(r"C:\Users\byz\Desktop\cebx\ItemList.txt", "wb")
    decompressobj = zlib.decompressobj()
    data = f.read(length)
    endFile.write(decompressobj.decompress(data))
    endFile.write(decompressobj.flush())
    print("解压缩ItemList完成")

def readEntry(current, f, bp):
    # 跳过Entry标识C.en 4字节
    current += 4
    f.seek(current)
    # 读入Entry长度 4字节
    EntryLength = f.read(4)
    current += 4
    print(EntryLength)
    EL = struct.unpack("<I", EntryLength)
    EL = list(EL)
    print("Entry长度为：" + str(EL[0]))
    # 读入Entry偏移量=bp
    forma2 = "<" + transformBP(bp)
    f.seek(current)
    eo = f.read(bp)
    current += bp
    print(eo)
    EO = struct.unpack(forma2, eo)
    EO = list(EO)
    # 文件流偏移位置
    bo = EO[0]
    print("Entry偏移量：" + str(EO[0]))
    # 读入下个Entry偏移=bp
    f.seek(current)
    nex = f.read(bp)
    current += bp
    print(nex)
    NEX = struct.unpack(forma2, nex)
    NEX = list(NEX)
    nextE = NEX[0]
    print("下个Entry偏移：" + str(NEX[0]))
    # 读入Compress值  1字节
    f.seek(current)
    com = f.read(1)
    current += 1
    print(com)
    COM = struct.unpack("<B", com)
    COM = list(COM)
    print("Compress:" + str(COM[0]) + "  (03为双zlib压缩 01为NameTable Zlib 02为itemList Zlib)")
    # 读入校验码 16字节
    f.seek(current)
    check = f.read(16)
    current += 16
    print(check)
    # CHECK = struct.unpack("<16B", check)
    # CHECK= list(CHECK)
    # print("MD5:" + str(CHECK[0]))
    # 读入路径名映射表长度 4字节
    f.seek(current)
    el = f.read(4)
    current += 4
    print(el)
    EL = struct.unpack("<I", el)
    EL = list(EL)
    elength = EL[0]
    print("路径名映射表长度(压缩后）:" + str(EL[0]))
    # 读zlib压缩TableName中的compressed data 并写出 1+1+3325+4=3331
    # current += 2
    # 解压nametable
    if COM[0]==1 | COM[0]==3:
        decompressNameTable(f,current,elength)
    current += elength
    ilength=EL[0]-elength-29-2*bp
    # 解压itemlist
    if COM[0]==2 | COM[0]==3:
        decompressNameTable(f,current,ilength)

    # 打开bitStream
    current = bo
    f.seek(current)
    current += 4
    sign = f.read(4)
    print(sign)
    # sig = struct.unpack("<4B", sign)
    # sig = list(sig)
    # # checksum
    # f.seek(current)
    # current += 1
    # fcs = f.read(1)
    # print(fcs)
    # # FCS = struct.unpack("<B", fcs)
    # # FCS = list(FCS)
    # # print(FCS)
    # # fileLength文件流长度
    # f.seek(current)
    # current += bp
    # fl = f.read(bp)
    # FL = struct.unpack("<I", fl)
    # FL = list(FL)
    # print(FL)
    # # 压缩方法 02为zlib压缩 以0xFF结尾
    # f.seek(current)
    # current += 2
    # cm = f.read(2)
    # print(cm)
    # CM = struct.unpack("2B", cm)
    # CM = list(CM)
    # print(CM)
    # 处理FileStream

    # 第一遍跑会报错 不用在意，就是跑完了全部的文件流，然后把这个循环注释掉，再跑一遍main函数即可
    while True:
        current=BitStreamRW(current,f,bp)
    # 返回下一个entry偏移，为0就代表没有下一个，结束，跳出循环
    return nextE


# 主函数
def main():
    # cebx文件路径
    cebxFilePath = r"C:\Users\byz\Desktop\cebx\shouce.cebx"
    with open(cebxFilePath, mode="rb") as f:
        # 解析数据包头文件 (
        # 当前文件流下标（字节）
        current = 16
        # 跳过版权信息及版本号（固定14+1+1=16字节）
        f.seek(current)
        # 读入Entry个数 4个字节
        EntryCount = f.read(4)
        # 改变下标
        current += len(EntryCount)
        # 打印entrycount数据，以 16 进制数显示
        print(EntryCount)
        # 字节解析为unsigned int 数据 小端寻址
        EC = struct.unpack("<I", EntryCount)
        # 将元组转为 list
        EC = list(EC)
        # entry包个数
        entrys = EC[0]
        print("Entry个数为：" + str(EC[0]))
        # 跳过EntryNameTabelType 默认取值0x00
        current += 1
        f.seek(current)
        # 读入BitsParam值 1字节
        BitsParam = f.read(1)
        print(BitsParam)
        # 改变下标
        current += 1
        # 解析打印
        BP = struct.unpack("<B", BitsParam)
        BP = list(BP)
        # 偏移
        bitsParam = BP[0]
        print("BitsParam值为：" + str(bitsParam))
        f.seek(current)
        # 读入头文件偏移量=bp
        feos = f.read(bitsParam)
        current += bitsParam
        print(feos)
        forma1 = "<" + transformBP(bitsParam)
        FEOS = struct.unpack(forma1, feos)
        FEOS = list(FEOS)
        print("头文件偏移量为：" + str(FEOS[0]))
        # END)
        # Entry部分 current=0代表没有下一个包了
        while current != 0:
            current = readEntry(current, f, bitsParam)
            print(current)
        # 第一遍跑可以注释掉以下部分，第二次再跑
        # 确认文件后缀名
        labelFile()
        # 解压图像数据包
        decompressIFC()
        # 解压字体数据包
        decompressEFC()
        # 获取bsg结点文件信息
        BSGAnaylize()
        # 还原基础xml文件
        dataMapAnaylize()


# 主函数入口 通用化方法
if __name__ == '__main__':
    main()

