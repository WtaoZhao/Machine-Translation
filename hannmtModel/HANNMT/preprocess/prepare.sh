#!/bin/bash
BASEDIR=$(pwd)
MOSES_SCRIPTS=$BASEDIR/../../mosesdecoder/scripts
DATA_SRC=$BASEDIR/src
DATA_DEST=$BASEDIR/dest
src=en
tgt=zh

for type in train valid;
do
  # Tokenize the English part
  cat $DATA_SRC/$type/$type.corpus.$src | \
  $MOSES_SCRIPTS/tokenizer/normalize-punctuation.perl -l $src | \
  $MOSES_SCRIPTS/tokenizer/tokenizer.perl -a -l $src  \
  > $DATA_DEST/$type/$type.corpus.tok.$src

  # Train truecaser and truecase
  $MOSES_SCRIPTS/recaser/train-truecaser.perl -model $DATA_DEST/truecase-model.$src -corpus $DATA_DEST/$type/$type.corpus.tok.$src
  $MOSES_SCRIPTS/recaser/truecase.perl < $DATA_DEST/$type/$type.corpus.tok.$src > $DATA_DEST/$type/$type.corpus.tc.$src -model $DATA_DEST/truecase-model.$src

  # Segment the Chinese part
  python -m jieba -d ' ' < $DATA_SRC/$type/$type.corpus.$tgt > $DATA_DEST/$type/$type.corpus.tok.$tgt

  ln -s $DATA_DEST/$type/$type.corpus.tok.$tgt  $DATA_DEST/$type/$type.corpus.tc.$tgt
done

python $BASEDIR/../source/preprocess.py -train_src $DATA_DEST/train/train.corpus.tc.$src -train_tgt $DATA_DEST/train/train.corpus.tc.$tgt -train_doc $DATA_SRC/train/train.corpus.doc -valid_src $DATA_DEST/valid/valid.corpus.tc.$src -valid_tgt $DATA_DEST/valid/valid.corpus.tc.$tgt -valid_doc $DATA_SRC/valid/valid.corpus.doc -save_data $BASEDIR/dataset/dataset -src_vocab_size 100000 -tgt_vocab_size 100000 -src_seq_length 100 -tgt_seq_length 100
