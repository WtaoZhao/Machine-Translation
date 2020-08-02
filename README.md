## 面向段落对齐语料的层级注意力翻译模型
### 1. 收集段对齐语料
> （1）在 Amazon 官网下载英汉双语小说电子版

> （2）将电子版小说转换为文本

> （3）结合程序进行人工校正对齐

> （4）整理得到段对齐语料库，拆分为训练集、验证集、测试集（/sentAlignProcess/src/）

> 注：数据集中也包含收集到的其它类型的段对齐语料，如[WIT语料库](https://wit3.fbk.eu/mt.php?release=2015-01)
### 2. 将段对齐语料转化为句对齐语料（sentAlignProcess）
> （1）利用 hardcut.py 将 /src 中段对齐语料库直接按标点拆开，并作相应标记，放入 /dest 中

> （2）利用 [Champollion-1.2](https://sourceforge.net/projects/champollion/)开源句对齐工具包将 /dest 中文件生成粗略句对齐索引文件，放入 /afterChamp 中

```
/home/[your DIR]/champollion-1.2/bin/champollion.EC_utf8 train[valid/test].cut.en train[valid/test].cut.zh train[valid/test].cut.align
```

> （3）利用 generate.py 中算法整理得到标准句对齐语料和段落索引文件，放入 /corpus 中

> 注：在实际实验过程中，我们发现 Champollion 无法处理过大文件（此处的训练集）的对齐，所以我们考虑将训练集按段落拆分成1000个小文件（详见/sentAlignProcess/trainSetMap.py），然后分别用 Champollion 对齐（详见drive.sh），再合并至一个对齐文件 train.cut.align（详见trainSetReduce.py）。
### 3. 数据集预处理
### 4. 训练模型
### 5. 测试模型
