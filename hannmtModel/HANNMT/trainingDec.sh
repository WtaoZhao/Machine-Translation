#!/bin/bash
# Training HAN-decoder using the sentence-level NMT model:
BASEDIR=$(pwd)
python $BASEDIR/source/train.py -data $BASEDIR/preprocess/dataset/dataset -save_model $BASEDIR/savedModel/HAN_dec/model -encoder_type transformer -decoder_type transformer -enc_layers 6 -dec_layers 6 -label_smoothing 0.1 -src_word_vec_size 512 -tgt_word_vec_size 512 -rnn_size 512 -position_encoding -dropout 0.1 -batch_size 1024 -start_decay_at 2 -report_every 100 -epochs 5 -max_generator_batches 32 -batch_type tokens -normalization tokens -accum_count 4 -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 1 -max_grad_norm 0 -param_init 0 -param_init_glorot -train_part all -context_type HAN_dec -context_size 3  -gpuid 3 #-train_from $BASEDIR/savedModel/sentence_level/model_acc_14.74_ppl_432.09_e1.pt
# Input options:
# - train_part:	[sentences, context, all]
# - context_type:	[HAN_enc, HAN_dec, HAN_join, HAN_dec_source, HAN_dec_context]
# - context_size:	number of previous sentences
# NOTE: The transformer model is sensitive to variation on hyperparameters. The HAN is also sensitive to the batch size.
