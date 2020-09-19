## 面向段落对齐语料的层级注意力翻译模型
#### Neural Machine Translation with Hierarchical Attention Networks based on Paragraph-parallel Corpus
![flowchart](https://github.com/Nick-Zhao-Engr/Machine-Translation/blob/master/images/flowchart.png)
### 1. 收集段对齐语料
> （1）在 [Amazon 中国](https://www.amazon.cn)官网下载英汉双语小说电子版；

> （2）将电子版小说转换为文本；

> （3）结合程序进行人工校正对齐；

> （4）整理得到段对齐语料库，拆分为训练集、验证集、测试集（/sentAlignProcess/src/）。

> 注：数据集中也包含收集到的其它类型的段对齐语料，如[WIT语料库](https://wit3.fbk.eu/mt.php?release=2015-01)。
### 2. 段落分割：将段对齐语料转化为句对齐语料
> （1）利用 hardcut.py 将 /src 中段对齐语料库直接按标点拆开，并作相应标记，放入 /dest 中；

> （2）利用 [Champollion-1.2](https://sourceforge.net/projects/champollion/) 开源句对齐工具包将 /dest 中文件生成粗略句对齐索引文件，放入 /afterChamp 中，该对齐工具基于论文 [\[1\]](https://www.cs.brandeis.edu/~marc/misc/proceedings/lrec-2006/pdf/746_pdf.pdf) ；

```
/home/[your DIR]/champollion-1.2/bin/champollion.EC_utf8 train[valid/test].cut.en train[valid/test].cut.zh train[valid/test].cut.align
```

> （3）利用 generate.py 中算法整理得到标准句对齐语料和段落索引文件，放入 /corpus 中。

> 注1：在实际实验过程中，我们发现 Champollion 无法处理过大文件（此处的训练集）的对齐，所以我们考虑将训练集按段落拆分成1000个小文件（详见trainSetMap.py），然后分别用 Champollion 对齐（详见drive.sh），再合并至一个对齐文件 train.cut.align（详见trainSetReduce.py）；

> 注2：.doc 段落索引文件含义详见 [idiap/HAN_NMT](https://github.com/idiap/HAN_NMT#preprocess)。
### 3. 数据集预处理
> 将上一步骤中得到的语料放入 /hannmtModel/HANNMT/preprocess/src 中，调用脚本 prepare.sh 进行预处理，对训练集和验证集分别进行英语标准化和汉语分词，得到预处理好的数据集（/hannmtModel/HANNMT/preprocess/dataset）。

> 注：第3、4、5部分参考了 [idiap/HAN_NMT](https://github.com/idiap/HAN_NMT) 的代码，是论文 [\[2\]](https://arxiv.org/abs/1809.01576) 中提到的篇章级层级注意力网络的 OpenNMT-Pytorch 实现。我们对原代码进行修改，使之更适合段落级的翻译，其中英语标准化使用 [moses](http://www.statmt.org/moses/) 工具，中文分词使用 [jieba](https://github.com/fxsjy/jieba) Python 库。此部分代码运行要求在 /hannmtModel 目录下安装 moses。
### 4. 训练模型
> 运行 shell 脚本直接训练模型，模型超参数需要在脚本中调节，包含以下四个脚本：

> trainingBase.sh 训练句级 Transformer 基准模型

> trainingEnc.sh 训练层级编码器（基于句级）

> trainingDec.sh 训练层级解码器（基于句级）

> trainingJoint.sh 训练层级联合模型（基于层级编码器、解码器）

> 每训练 1 个 epoch ， 模型 checkpoint 自动保存，有助于预训练或更换数据集进一步训练。
### 5. 测试模型
> 将第 2 部分生成的测试语料用于此步的模型测试，尝试将英文翻译成中文。测试环节分以下几个步骤：

> （1）运行脚本 translate.sh 对测试语料中的英文进行翻译，生成翻译原始文件 prediction.raw.zh ；

> （2）对上述原始文件，调用 process.py 处理，生成翻译句子级文件 prediction.sent.zh 和段落级文件 prediction.para.zh ；

> （3）计算 BLEU 分数 [\[3\]](https://www.aclweb.org/anthology/P02-1040.pdf)。直接调用 calculateBLEU.py 进行计算，输出 1,2,3,4-BLEU 分数，对翻译结果的充分性和流畅性进行评估，也便于与其它翻译模型进行比较。

> 说明：在计算 BLEU 分数时，将句子级预测 prediction.sent.zh 作为待评估文件（candidate），将测试语料目标文件 test.corpus.zh 、测试语料源文件 test.corpus.en 的 Google/Baidu 翻译结果三者作为参考文件（Reference），以便计算得到更客观合理的 BLEU 分数。
### 6. 模型对比与评估
> 将第 2 部分生成的语料库作为以下翻译模型的输入，进行训练、测试、计算 BLEU 分数，对比评估本文提出的模型相对于前人翻译模型的优势与不足，代码位于 /hannmtModel/HANNMT/compare/ ，Pytorch 实现，参考代码 [bentrevett/pytorch-seq2seq](https://github.com/bentrevett/pytorch-seq2seq)。

> 1 - RNN Encoder-Decoder for SMT [\[code\]](https://github.com/Nick-Zhao-Engr/Machine-Translation/blob/master/hannmtModel/HANNMT/compare/1_RNN_Encoder-Decoder_for_SMT.py) (parallel-gpu)

[Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation, 2014/9](https://arxiv.org/pdf/1406.1078.pdf)

> 2 - LSTM Encoder-Decoder for NMT [\[code\]](https://github.com/Nick-Zhao-Engr/Machine-Translation/blob/master/hannmtModel/HANNMT/compare/2_LSTM_Encoder-Decoder_for_NMT.py) (parallel-gpu)

[Sequence to Sequence Learning with Neural Networks, 2014/12](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)

> 3 - RNN Encoder-Decoder with Attention Mechanism [\[code\]](https://github.com/Nick-Zhao-Engr/Machine-Translation/blob/master/hannmtModel/HANNMT/compare/3_RNN_Encoder-Decoder_with_Attention_Mechanism.py) (parallel-gpu)

[Neural Machine Translation by Jointly Learning to Align and Translate, 2016/5](https://arxiv.org/pdf/1409.0473.pdf)

> 4 - ConvS2S for NMT [\[code\]](https://github.com/Nick-Zhao-Engr/Machine-Translation/blob/master/hannmtModel/HANNMT/compare/4_ConvS2S_for_NMT.py) (parallel-gpu)

[Convolutional Sequence to Sequence Learning, 2017/7](https://arxiv.org/pdf/1705.03122.pdf)

> 5 - Transformer model [\[code\]](https://github.com/Nick-Zhao-Engr/Machine-Translation/blob/master/hannmtModel/HANNMT/compare/5_Transformer_model.py) (parallel-gpu)

[Attention Is All You Need, 2017/12](http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) (i.e. sentence-level HAN)
### 参考文献
> [1] Ma, Xiaoyi. "Champollion: A Robust Parallel Text Sentence Aligner." LREC. 2006.

> [2] Miculicich, Lesly, et al. "Document-level neural machine translation with hierarchical attention networks." arXiv preprint arXiv:1809.01576 (2018).

> [3] Papineni, Kishore, et al. "BLEU: a method for automatic evaluation of machine translation." Proceedings of the 40th annual meeting of the Association for Computational Linguistics. 2002.
