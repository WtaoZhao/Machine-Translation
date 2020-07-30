'''使用方法：见main函数'''
from nltk.tokenize import sent_tokenize,word_tokenize
import re
import jieba

def cut_ch_sent(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")

def ch_sentence_number(content):
    '''
    content是read的结果
    返回一个列表res, res[i]表示第i个单词所在的句子的序号
    se_ind从0开始, word_ind也是
    '''
    se_list = cut_ch_sent(content)
    word_list = ch_cut_word(content)
    se_ind=0
    cnt=0
    res=[]
    if(len(se_list)==0):
        raise ValueError('number of sentences is 0"')
    se_len=len(se_list[0])
    for word_ind in range(len(word_list)):
        cnt+=len(word_list[word_ind])
        if cnt>se_len:
            se_ind+=1
            se_len+=len(se_list[se_ind])
        res.append(se_ind)
    return res

def ch_cut_word(se):
    # 返回列表
    cut_sen = jieba.lcut(se)
    return cut_sen

def cut_en_sentence(content):
    '''英文分句，返回列表'''
    return sent_tokenize(content)

def en_cut_word(content):
    '''返回列表'''
    return word_tokenize(content)

def en_sentence_number(content):
    word_list=word_tokenize(content)
    se_list=cut_en_sentence(content)
    se_ind = 0
    cnt = 0
    res = []
    if (len(se_list) == 0):
        raise ValueError('number of sentences is 0"')
    se_len = len(se_list[0])
    for word_ind in range(len(word_list)):
        cnt += len(word_list[word_ind])
        if cnt > se_len:
            se_ind += 1
            se_len += len(se_list[se_ind])
        res.append(se_ind)
    return res

def maxelements(seq):
    ''' Return list of position(s) of largest element '''
    max_indices = []
    if seq:
        max_val = seq[0]
        for i,val in ((i,val) for i,val in enumerate(seq) if val >= max_val):
            if val == max_val:
                max_indices.append(i)
            else:
                max_val = val
                max_indices = [i]

    return max_indices

def match(ch_ind,en_ind,ch_word_sentence_number,en_word_sentence_number,file):
    '''
    ch_ind 与 en_ind 是配对的单词序号，ch_word_sentence_number[i]表示第i个单词所在的句子序号
    对句子进行配对，答案写入 file
    '''
    ch_sentence_number=ch_word_sentence_number[-1]+1  # 中文句子的数量
    en_sentence_number=en_word_sentence_number[-1]+1
    print('ch sen number:{} en sen number:{}'.format(ch_sentence_number,en_sentence_number))
    ch_mat=[[0]*en_sentence_number for _ in range(ch_sentence_number)]  # 记录ch_mat[i][j]表示第i个中文句子与第j个英文句子配对的可能性（为整数）
    write_ans=open(file,'w',encoding='utf-8')
    for i in range(len(ch_ind)):
        ch_word_ind=ch_ind[i]
        en_word_ind=en_ind[i]
        # print('{} {}\n'.format(ch_word_ind, en_word_ind))
        en_sentence_ind=en_word_sentence_number[en_word_ind]
        ch_sentence_ind=ch_word_sentence_number[ch_word_ind]

        ch_mat[ch_sentence_ind][en_sentence_ind]+=1
    print(ch_mat)
    for i in range(ch_sentence_number):
        prob_distribution=ch_mat[i]
        max_indices=maxelements(prob_distribution)
        write_ans.write('{}-{}\n'.format(i,max_indices[0]))

def cut_ch_and_en(read_ch,read_en,ch_process_file='ch.txt',en_process_file='en.txt'):
    '''中文分词、英文不作处理，用来给GIZA++做预处理
    read_ch和read_en是中英文原文件（未分词）的读取结果
    ch_process_file是存储中文分词的文件，en_process_file是英文标准化之后的文件'''
    ch_word_list = ch_cut_word(read_ch)
    ch_processed = ' '.join(ch_word_list)
    ch_write = open(ch_process_file, 'w', encoding='utf-8')
    ch_write.write(ch_processed)

    en_word_list = en_cut_word(read_en)
    en_processed = ' '.join(en_word_list)
    en_write = open(en_process_file, 'w', encoding='utf-8')
    en_write.write(en_processed)

def preprocess(content,read_ch,read_en,translation_file='match_translation2.txt'):
    '''content是读取双向对齐文件的结果，现在只能处理一段。
    read_ch和read_en是中英文原文件（未分词）的读取结果
    translation_file: 把对齐的结果从数字翻译成单词
    得到ch_word_list, en_word_list,ch_ind,en_ind,ch_word_sentence_number,en_word_sentence_number
    ch_ind 与 en_ind 是配对的单词序号，
    ch_word_sentence_number[i]表示第i个单词所在的句子序号
    '''
    content_list = content.split()
    ch_word_list = ch_cut_word(read_ch)
    en_word_list = en_cut_word(read_en)

    ch_ind = []  # ch_ind[i]与en_ind[i]配对，ch_ind[i]是单词的序号
    en_ind = []
    write_match = open(translation_file, 'w', encoding='utf-8')  # 把对齐的结果从数字翻译成单词
    for pair in content_list:
        li = pair.split('-')
        ch = ch_word_list[int(li[0]) - 1]
        en = en_word_list[int(li[1]) - 1]
        write_match.write('{}-{}\n'.format(ch, en))
        ch_ind.append(int(li[0]) - 1)
        en_ind.append(int(li[1]) - 1)

    ch_word_sentence_number = ch_sentence_number(read_ch)
    en_word_sentence_number = en_sentence_number(read_en)

    return ch_word_list,en_word_list,ch_ind,en_ind,ch_word_sentence_number,en_word_sentence_number

if __name__=='__main__':
    read_match_res = open('res.txt','r',encoding='utf-8')
    match_res = read_match_res.read()
    read_ch = open('ch_src.txt','r',encoding='utf-8')
    ch = read_ch.read()
    read_en = open('en_src.txt', 'r', encoding='utf-8')
    en = read_en.read()
    ch_word_list, en_word_list, ch_ind, en_ind, \
    ch_word_sentence_number, en_word_sentence_number = preprocess(match_res,ch,en)
    # print('ch word num:{} en word num:{}'.format(len(ch_word_list),len(en_word_list)))
    # print('ch_word_sentence_number\'s len:{} {}'.format(len(ch_word_sentence_number),len(en_word_sentence_number)))
    sent_match_res_file = '句子对齐.txt'
    match(ch_ind,en_ind,ch_word_sentence_number,en_word_sentence_number,sent_match_res_file)








