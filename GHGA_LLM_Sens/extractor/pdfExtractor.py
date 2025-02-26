import pdfplumber

with pdfplumber.open(r"D:\bishesystem\file\1.pdf") as pdf:
    for page in pdf.pages:
        print(page.extract_text())

        # 每页打印一分页分隔
        print('---------- 分页分隔 ----------')
        break
