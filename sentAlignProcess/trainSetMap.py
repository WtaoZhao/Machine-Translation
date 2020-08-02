train_en = open('./dest/train.cut.en', 'r', encoding='utf-8').readlines()
train_zh = open('./dest/train.cut.zh', 'r', encoding='utf-8').readlines()

nums = 1000
paraNums = 0

for sent in train_en:
    if sent == '#\n':
        paraNums += 1

paraPerFile = int(paraNums / nums)
paraLastFile = paraNums - paraPerFile * (nums - 1)

setIdx = open('./trainSetSplit/split.idx', 'w', encoding='utf-8')
enIdx = 0
zhIdx = 0
paraCnt = 0

for i in range(nums - 1):
    en = open('./trainSetSplit/train.%d.en' % i, 'w', encoding='utf-8')
    zh = open('./trainSetSplit/train.%d.zh' % i, 'w', encoding='utf-8')

    setIdx.write(str(enIdx) + '\t')
    while paraCnt < paraPerFile:
        en.write(train_en[enIdx])
        if train_en[enIdx] == '#\n':
            paraCnt += 1
        enIdx += 1

    setIdx.write(str(zhIdx) + '\n')
    while paraCnt > 0:
        zh.write(train_zh[zhIdx])
        if train_zh[zhIdx] == '#\n':
            paraCnt -= 1
        zhIdx += 1

    en.close()
    zh.close()

en = open('./trainSetSplit/train.%d.en' % (nums - 1), 'w', encoding='utf-8')
zh = open('./trainSetSplit/train.%d.zh' % (nums - 1), 'w', encoding='utf-8')
train_en_len = len(train_en)
train_zh_len = len(train_zh)

setIdx.write(str(enIdx) + '\t' + str(zhIdx) + '\n')
for i in range(enIdx, train_en_len):
    en.write(train_en[i])
for i in range(zhIdx, train_zh_len):
    zh.write(train_zh[i])

en.close()
zh.close()
