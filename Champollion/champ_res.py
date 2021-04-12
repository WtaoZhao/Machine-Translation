def align_sent(ch_file, en_file, res_file, write_ch, write_en):
    '''chfile, enfile是中文、英文的语料，一句话一段，
    res_file是champollion对齐的结果，该函数把没有匹配到的
    中英文句子删掉，如果出现一个ommit, 前一句话和后一句话都要删掉
    删除部分内容后写入write_ch和write_en文件里面，格式是一句话为一段'''
    read_ch = open(ch_file,'r',encoding='utf-8')   # 编码若出问题，需要做出调整
    read_en = open(en_file, 'r', encoding='utf-8')
    read_res = open(res_file,'r',encoding='utf-8')

    wch = open(write_ch,'w',encoding='utf-8')
    wen = open(write_en,'w',encoding='utf-8')
    ch_cnt,en_cnt = 1,1  # 从1开始，读取的第一句话序号记为1
    ch_sent,en_sent='',''
    write_cnt=0
    while True:
        pair = read_res.readline()
        if pair =='':
            break  # 读取结束
        if pair.find('<=>') != -1:
            pair_list = pair.split('<=>')
            first = pair_list[0].strip()
            second = pair_list[1].strip()
            print(first, second)
            if (first == 'omitted' and second != 'omitted')\
                    or (first!='omitted' and second != 'omitted'):
                print('flag1')
                second_list = second.split(',')
                new_ch_cnt =int(second_list[-1])
                for i in range(new_ch_cnt-ch_cnt+1):
                    ch_sent+=read_ch.readline().strip()
                    print(ch_sent)
                ch_cnt = new_ch_cnt+1
            if (first != 'omitted' and second == 'omitted')\
                    or (first!='omitted' and second != 'omitted'):
                print('flag2')
                first_list = first.split(',')
                new_en_cnt = int(first_list[-1])
                for i in range(new_en_cnt - en_cnt + 1):
                    en_sent += read_en.readline().strip()
                    print(en_sent)
                en_cnt = new_en_cnt+1
            if first!='omitted' and second != 'omitted':
                print('flag3')
                wch.write('{}\n'.format(ch_sent))
                wen.write('{}\n'.format(en_sent))

                write_cnt+=1
            ch_sent,en_sent='',''
    print(write_cnt)

if __name__=='__main__':
    ch_file, en_file = 'zh.txt', 'en.txt'
    res = 'res.txt'
    write_ch, write_en = 'aligned_zh.txt', 'aligned_en.txt'
    align_sent(ch_file, en_file, res, write_ch, write_en)
