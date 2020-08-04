#!/bin/bash
BASEDIR=$(pwd)
python $BASEDIR/source/translate.py -model $BASEDIR/savedModel/model_acc_22.27_ppl_239.71_e1.pt -src $BASEDIR/translateTest/srcData/test.corpus.en -doc $BASEDIR/translateTest/srcData/test.corpus.doc -output $BASEDIR/translateTest/prediction.raw.zh -translate_part all -batch_size 1000 -gpu 3
