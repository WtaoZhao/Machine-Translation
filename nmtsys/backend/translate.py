import argparse
import os
import sys
import torch
import jieba
import time

Tokenizer = './backend/script/tokenizer.perl'
FastBpe = './backend/script/fast'
BpeCode = './backend/model/codes_xnli_100'
Vocab ='./backend/model/vocab_xnli_100'

class Translator:
    def __init__(self, src_lg, trg_lg, model_path=None, device=None):
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        if model_path is None:
            model_path = './backend/model/NMT.{}-{}.pth'.format(src_lg, trg_lg)
        self.model = torch.load(model_path)
        self.src_lg = src_lg
        self.trg_lg = trg_lg

    def preprocess(self, sentences, encoding='utf8'):
        temp_filename = './backend/temp/{}-{}.temp'.format(self.src_lg, self.trg_lg)
        with open(temp_filename, 'w', encoding=encoding) as fout:
            for sentence in sentences:
                fout.write('{}\n'.format(sentence.strip()))
        # tokenize
        os.system('{} -l {} < {} > {}.tok'.format(Tokenizer, self.src_lg, temp_filename, temp_filename))
        # jieba
        if self.src_lg == 'zh':
            with open(temp_filename+'.tok', 'r', encoding=encoding) as fin:
                with open(temp_filename+'.tok.jieba', 'w', encoding=encoding) as fout:
                    for sentence in fin.readlines():
                        fout.write('{}\n'.format( ' '.join( jieba.cut( sentence.strip() ) ) ))
            os.system('mv {} {}'.format(temp_filename + '.tok.jieba', temp_filename + '.tok'))
        # bpe
        os.system('{} applybpe {}.bpe {} {} {}'.format(FastBpe, temp_filename, temp_filename + '.tok', BpeCode, Vocab))
        processed_sentences = []
        with open(temp_filename + '.bpe', 'r', encoding=encoding) as fin:
            for line in fin.readlines():
                processed_sentences.append(line.strip())
        return processed_sentences

    def remove_special_tokens(self, sentences, lg):
        for i in range(len(sentences)):
            sentences[i] = sentences[i].replace('@@ ', '')
            sentences[i] = sentences[i].replace('<OOV>', '')
            if lg == 'zh':
                sentences[i] = sentences[i].replace(' ', '')

    def translate(self, sentences, encoding='utf8', beam_size=0):
        if len(sentences) == 0:
            raise Exception()
        sentences = self.preprocess(sentences, encoding=encoding)
        if beam_size <= 0:
            translation = self.model.greedy(sentences, train=False)
        else:
            translation = self.model.beam_search(sentences, train=False, beam_size=beam_size)
        self.remove_special_tokens(translation, self.trg_lg)
        return translation

def demo():
    translator = Translator('ne', 'zh')
    start_time = time.time()
    print('Model is loaded.')
    sentences = [
        'डाटाबेससँग तस्विर मेटाडाटा सिन्क गर्नुहोस् । कृपया प्रतिक्षा गर्नुहोस्...',
        'एल्बम क्रमबद्ध गर्नुहोस्',
        'अभिमुखीकरण ट्याग अनुसार छवि/ थम्ब परिक्रमण देखाउनुहोस्',
        'परिक्रमण/ घुमाएपछि अभिमुखीकरण ट्यागलाई साधारणमा सेट गर्नुहोस्',
        'पूरा पर्दा मोडमा उपकरणपट्टी लुकाउनुहोस्'
    ]
    translation = translator.translate(sentences)
    print(translation)
    end_time = time.time()
    print('Time: ', end_time - start_time)


if __name__ == '__main__':
    demo()
