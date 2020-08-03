## 面向段落对齐语料的层级注意力翻译模型
### 1. 收集段对齐语料
> （1）在 Amazon 官网下载英汉双语小说电子版；

> （2）将电子版小说转换为文本；

> （3）结合程序进行人工校正对齐；

> （4）整理得到段对齐语料库，拆分为训练集、验证集、测试集（/sentAlignProcess/src/）。

> 注：数据集中也包含收集到的其它类型的段对齐语料，如[WIT语料库](https://wit3.fbk.eu/mt.php?release=2015-01)。
### 2. 将段对齐语料转化为句对齐语料（sentAlignProcess）
> （1）利用 hardcut.py 将 /src 中段对齐语料库直接按标点拆开，并作相应标记，放入 /dest 中；

> （2）利用 [Champollion-1.2](https://sourceforge.net/projects/champollion/) 开源句对齐工具包将 /dest 中文件生成粗略句对齐索引文件，放入 /afterChamp 中，该对齐工具基于论文 [1](https://www.cs.brandeis.edu/~marc/misc/proceedings/lrec-2006/pdf/746_pdf.pdf) ；

```
/home/[your DIR]/champollion-1.2/bin/champollion.EC_utf8 train[valid/test].cut.en train[valid/test].cut.zh train[valid/test].cut.align
```

> （3）利用 generate.py 中算法整理得到标准句对齐语料和段落索引文件，放入 /corpus 中。

> 注1：在实际实验过程中，我们发现 Champollion 无法处理过大文件（此处的训练集）的对齐，所以我们考虑将训练集按段落拆分成1000个小文件（详见trainSetMap.py），然后分别用 Champollion 对齐（详见drive.sh），再合并至一个对齐文件 train.cut.align（详见trainSetReduce.py）；
> 注2：.doc 段落索引文件含义详见 [idiap/HAN_NMT](https://github.com/idiap/HAN_NMT#preprocess)
### 3. 数据集预处理
> 将上一步骤中得到的语料放入 /hannmtModel/HANNMT/preprocess/src 中，调用脚本 prepare.sh 进行预处理，对训练集和验证集分别进行英语标准化和汉语分词，得到预处理好的数据集（/hannmtModel/HANNMT/preprocess/dataset）。
> 注：第3、4、5部分参考了 [idiap/HAN_NMT](https://github.com/idiap/HAN_NMT) 的代码，是论文 [2](https://arxiv.org/abs/1809.01576) 中提到的篇章级层级注意力网络的 OpenNMT-Pytorch 实现。我们对原代码进行修改，使之更适合段落级的翻译，其中英语标准化使用 [moses](http://www.statmt.org/moses/) 工具，中文分词使用 [jieba](https://github.com/fxsjy/jieba) Python 库。此部分代码运行要求在 /hannmtModel 目录下安装 moses。
### 4. 训练模型
### 5. 测试模型
### 参考文献
> [1] Ma, Xiaoyi. "Champollion: A Robust Parallel Text Sentence Aligner." LREC. 2006.
> [2] Miculicich, Lesly, et al. "Document-level neural machine translation with hierarchical attention networks." arXiv preprint arXiv:1809.01576 (2018).
