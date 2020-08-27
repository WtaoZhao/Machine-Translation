def to_tsv(en_file, zh_file, out_file):
    en = open(en_file, 'r', encoding='utf-8')
    zh = open(zh_file, 'r', encoding='utf-8')
    out = open(out_file, 'w', encoding='utf-8')

    zh_lines = zh.readlines()
    en_lines = en.readlines()

    for i in range(len(zh_lines)):
        out.write(en_lines[i].strip('\n').replace('\t', ''))
        out.write('\t')
        out.write(zh_lines[i].replace('\t', ''))

if __name__ == "__main__":
    BASEDIR = '../../preprocess/src/'
    for type in ['train', 'valid', 'test']:
        en_file = BASEDIR + type + '/' + type + '.corpus.en'
        zh_file = BASEDIR + type + '/' + type + '.corpus.zh'
        out_file = type + '.tsv'
        to_tsv(en_file, zh_file, out_file)
