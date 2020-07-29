from res_process import ch_cut_word, en_cut_word

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

if __name__=='__main__':
    read_ch = open('ch_src.txt', 'r', encoding='utf-8')
    ch = read_ch.read()
    read_en = open('en_src.txt', 'r', encoding='utf-8')
    en = read_en.read()

    ch_file = 'ch.txt'
    en_file = 'en.txt'
    cut_ch_and_en(ch,en,ch_file,en_file)
