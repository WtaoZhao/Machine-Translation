import os

for type in ['train', 'valid', 'test']:
    en_dest = open('./dest/%s.cut.en' % type, 'r', encoding='utf-8').readlines()
    zh_dest = open('./dest/%s.cut.zh' % type, 'r', encoding='utf-8').readlines()
    align = open('./afterChamp/%s.cut.align' % type, 'r', encoding='utf-8').readlines()
    en_out = open('./corpus/%s.corpus.en.tmp' % type, 'w+', encoding='utf-8')
    zh_out = open('./corpus/%s.corpus.zh.tmp' % type, 'w+', encoding='utf-8')
    en_dest_len = len(en_dest)
    zh_dest_len = len(zh_dest)
    enFlag = [True]
    zhFlag = [True]

    def writelines(left_nums, right_nums, enFlag, zhFlag):
        '''左边为 en / 右边为 zh'''

        # 如果两边都只有 '#'
        if left_nums and right_nums:
            if len(left_nums) == 1 and len(right_nums) == 1:
                if en_dest[left_nums[0] - 1] == '#\n' and zh_dest[right_nums[0] - 1] == '#\n':
                    if enFlag[0]:
                        en_out.write('#\n')
                    else:
                        en_out.write('\n#\n')
                    if zhFlag[0]:
                        zh_out.write('#\n')
                    else:
                        zh_out.write('\n#\n')
                    return

        # 异常处理
        if left_nums:
            for num in left_nums:
                if '#' in en_dest[num - 1]:
                    return
        if right_nums:
            for num in right_nums:
                if '#' in zh_dest[num - 1]:
                    return

        enFlag[0] = True
        zhFlag[0] = True

        # 如果左边为 omitted
        if not left_nums:
            idx = right_nums[0] - 1
            if zh_dest[idx - 1] == '#\n' and zh_dest[idx + 1] == '#\n':
                return
            elif idx == 0 or zh_dest[idx - 1] == '#\n':
                zh_out.write(zh_dest[idx].rstrip('\n').replace('\t', ''))
                zhFlag[0] = False
            elif idx == zh_dest_len - 1 or zh_dest[idx + 1] == '#\n':
                zh_out.seek(zh_out.tell() - 2)
                zh_out.truncate()
                zh_out.seek(zh_out.tell() - 1)
                try:
                    if zh_out.read() == '#':
                        zh_out.write('\n')
                except UnicodeDecodeError:
                    pass
                zh_out.write(zh_dest[idx].replace('\t', ''))
            return

        # 如果右边为 omitted
        if not right_nums:
            idx = left_nums[0] - 1
            if en_dest[idx - 1] == '#\n' and en_dest[idx + 1] == '#\n':
                return
            if idx == 0 or en_dest[idx - 1] == '#\n':
                en_out.write(en_dest[idx].rstrip('\n').replace('\t', '') + ' ')
                enFlag[0] = False
            elif idx == en_dest_len - 1 or en_dest[idx + 1] == '#\n':
                en_out.seek(en_out.tell() - 2)
                en_out.truncate()
                en_out.seek(en_out.tell() - 1)
                if en_out.read() == '#':
                    en_out.write('\n')
                else:
                    en_out.write(' ')
                en_out.write(en_dest[idx].replace('\t', ''))
            return

        # 如果没有 omitted
        for i in range(len(left_nums)):
            idx = left_nums[i] - 1
            if i == len(left_nums) - 1:
                en_out.write(en_dest[idx].replace('\t', ''))
            else:
                en_out.write(en_dest[idx].rstrip('\n').replace('\t', '') + ' ')

        for i in range(len(right_nums)):
            idx = right_nums[i] - 1
            if i == len(right_nums) - 1:
                zh_out.write(zh_dest[idx].replace('\t', ''))
            else:
                zh_out.write(zh_dest[idx].rstrip('\n').replace('\t', ''))



    # 处理过程
    for line in align:
        mid = line.find(' <=> ')
        left = line[ : mid]
        right = line[mid + 5 : -1]
        left_nums = []
        right_nums = []

        # 存在 omitted
        if left == 'omitted' or right == 'omitted':
            if left == 'omitted':
                right_nums.append(int(right))
            else:
                left_nums.append(int(left))

        # 不存在 omitted
        else:
            tmp_left = left.split(',')
            tmp_right = right.split(',')
            for elem in tmp_left:
                left_nums.append(int(elem))
            for elem in tmp_right:
                right_nums.append(int(elem))

        writelines(left_nums, right_nums, enFlag, zhFlag)


    en_out.close()
    zh_out.close()
    en_out = open('./corpus/%s.corpus.en.tmp' % type, 'r', encoding='utf-8').readlines()

    with open('./corpus/%s.corpus.zh.tmp' % type, 'rb') as zh_out:
        with open('./corpus/%s.corpus.zh.utf8tmp' % type, "w", encoding="utf-8") as f:
            for line in zh_out:
                if not line:
                    break
                else:
                    line = line.decode("utf-8", "ignore")
                    f.write(str(line).rstrip() + '\n')

    zh_out = open('./corpus/%s.corpus.zh.utf8tmp' % type, 'r', encoding='utf-8').readlines()
    en = open('./corpus/%s.corpus.en' % type, 'w', encoding='utf-8')
    zh = open('./corpus/%s.corpus.zh' % type, 'w', encoding='utf-8')
    doc = open('./corpus/%s.corpus.doc' % type, 'w', encoding='utf-8')

    cnt = 0
    en_idx = 0
    zh_idx = 0
    en_len = len(en_out)
    zh_len = len(zh_out)

    while en_idx < en_len and zh_idx < zh_len:
        tmp_en = []
        tmp_zh = []
        while en_idx < en_len and en_out[en_idx] != '#\n':
            tmp_en.append(en_out[en_idx])
            en_idx += 1
        en_idx += 1
        while zh_idx < zh_len and zh_out[zh_idx] != '#\n':
            tmp_zh.append(zh_out[zh_idx])
            zh_idx += 1
        zh_idx += 1
        if len(tmp_en) == len(tmp_zh):
            doc.write(str(cnt) + '\n')
            cnt += len(tmp_en)
            for i in range(len(tmp_en)):
                en.write(tmp_en[i])
                zh.write(tmp_zh[i])

    en.close()
    zh.close()
    doc.close()
    os.remove('./corpus/%s.corpus.en.tmp' % type)
    os.remove('./corpus/%s.corpus.zh.tmp' % type)
    os.remove('./corpus/%s.corpus.zh.utf8tmp' % type)
