from nltk.tokenize import sent_tokenize
import re

for type in ['train', 'valid', 'test']:
    en_src = open('./src/%s.en' % type, 'r', encoding='utf-8').readlines()
    zh_src = open('./src/%s.zh' % type, 'r', encoding='utf-8').readlines()
    en_dest = open('./dest/%s.cut.en' % type, 'w', encoding='utf-8')
    zh_dest = open('./dest/%s.cut.zh' % type, 'w', encoding='utf-8')

    '''英文分句'''
    for para in en_src:
        cut_list = sent_tokenize(para)
        for sent in cut_list:
            en_dest.write(sent + '\n')
        en_dest.write('#\n')

    en_dest.close()

    '''中文分句'''
    def cut_zh(para):
        para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
        para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
        para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
        para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
        # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
        para = para.rstrip()  # 段尾如果有多余的\n就去掉它
        # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
        return para.split("\n")

    for para in zh_src:
        cut_list = cut_zh(para)
        for sent in cut_list:
            zh_dest.write(sent + '\n')
        zh_dest.write('#\n')

    zh_dest.close()
