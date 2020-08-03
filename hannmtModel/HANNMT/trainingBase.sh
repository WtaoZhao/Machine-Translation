#!/bin/bash
# Training the sentence-level NMT baseline:
BASEDIR=$(pwd)
python $BASEDIR/source/train.py -data $BASEDIR/preprocess/dataset/dataset -save_model $BASEDIR/savedModel/sentence_level/model -encoder_type transformer -decoder_type transformer -enc_layers 6 -dec_layers 6 -label_smoothing 0.1 -src_word_vec_size 512 -tgt_word_vec_size 512 -rnn_size 512 -position_encoding -dropout 0.1 -batch_size 1024 -start_decay_at 20 -report_every 100 -epochs 10 -max_generator_batches 16 -batch_type tokens -normalization tokens -accum_count 4 -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 -max_grad_norm 0 -param_init 0 -param_init_glorot -gpuid 1 #-train_from $BASEDIR/savedModel/sentence_level/model_acc_14.74_ppl_432.09_e1.pt
