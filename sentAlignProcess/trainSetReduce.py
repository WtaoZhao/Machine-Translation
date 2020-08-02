setIdx = open('./trainSetSplit/split.idx', 'r', encoding='utf-8').readlines()
align = open('./afterChamp/train.cut.align', 'w', encoding='utf-8')
offset = [line.strip('\n').split('\t') for line in setIdx]
nums = len(offset)

for i in range(nums):
    align_tmp = open('./trainSetSplit/train.%d.align' % i, 'r', encoding='utf-8').readlines()
    for line in align_tmp:
        left, right = line.strip('\n').split(' <=> ')
        leftIdx = left.split(',')
        rightIdx = right.split(',')

        if leftIdx[0] != 'omitted':
            for j in range(leftIdx):
                leftIdx[j] = str(int(leftIdx[j]) + int(offset[i][0]))

        if rightIdx[0] != 'omitted':
            for j in range(rightIdx):
                rightIdx[j] = str(int(rightIdx[j]) + int(offset[i][1]))

        left = ','.join(leftIdx)
        right = ','.join(rightIdx)
        line = ' <=> '.join([left, right]) + '\n'
        align.write(line)
    align_tmp.close()

align.close()
