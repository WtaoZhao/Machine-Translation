import os

raw = open('prediction.raw.zh', 'r', encoding='utf-8').readlines()
doc = open('./srcData/test.corpus.doc', 'r', encoding='utf-8').readlines()
para = open('prediction.para.zh', 'w', encoding='utf-8')
sent = open('prediction.sent.zh', 'w', encoding='utf-8')
sentNums = len(raw)
cntPara = 0

'''生成段落级翻译文件'''
for i in range(sentNums):
    para.write(raw[i].strip('\n').replace(' ', ''))
    if i == int(doc[cntPara].strip('\n')):
        para.write('\n')
        cntPara += 1

'''生成句子级翻译文件'''
for i in range(sentNums):
    sent.write(raw[i].replace(' ', ''))

os.remove('prediction.raw.zh')
