import jieba
from nltk.translate.bleu_score import sentence_bleu

srcRef = open('./srcData/test.corpus.zh', 'r', encoding='utf-8').readlines()
googleRef = open('./otherRefData/googleTrans.zh', 'r', encoding='utf-8').readlines()
baiduRef = open('./otherRefData/baiduTrans.zh', 'r', encoding='utf-8').readlines()
sentCan = open('prediction.sent.zh', 'r', encoding='utf-8').readlines()

testlen = len(srcRef)
assert testlen == len(googleRef)
assert testlen == len(baiduRef)
assert testlen == len(sentCan)

bleu1 = 0.
bleu2 = 0.
bleu3 = 0.
bleu4 = 0.

print('Start calculating BLEU Score...')

for i in range(testlen):
    if i % 1000 == 0:
        print('[%d/%d]' % (i, testlen))
        print('Cumulative 1-gram: %f' % bleu1)
        print('Cumulative 2-gram: %f' % bleu2)
        print('Cumulative 3-gram: %f' % bleu3)
        print('Cumulative 4-gram: %f' % bleu4)

    srcRefList = jieba.lcut(srcRef[i].strip('\n'))
    googleRefList = jieba.lcut(googleRef[i].strip('\n'))
    baiduRefList = jieba.lcut(baiduRef[i].strip('\n'))

    ref = []
    ref.append(srcRefList)
    ref.append(googleRefList)
    ref.append(baiduRefList)
    can = jieba.lcut(sentCan[i].strip('\n'))

    b1 = sentence_bleu(ref, can, weights=(1., 0, 0, 0))
    b2 = sentence_bleu(ref, can, weights=(.5, .5, 0, 0))
    b3 = sentence_bleu(ref, can, weights=(.3333, .3333, .3333, 0))
    b4 = sentence_bleu(ref, can, weights=(.25, .25, .25, .25))

    bleu1 = (bleu1 * i + b1) / (i + 1.)
    bleu2 = (bleu2 * i + b2) / (i + 1.)
    bleu3 = (bleu3 * i + b3) / (i + 1.)
    bleu4 = (bleu4 * i + b4) / (i + 1.)

print('\nBLEU Score Percentage Form')
bleu1 *= 100
bleu2 *= 100
bleu3 *= 100
bleu4 *= 100
print('1-BLEU: %f' % bleu1)
print('2-BLEU: %f' % bleu2)
print('3-BLEU: %f' % bleu3)
print('4-BLEU: %f' % bleu4)
