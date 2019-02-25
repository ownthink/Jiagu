#!/usr/bin/env python3
# -*-coding:utf-8-*-
"""
 * Copyright (C) 2018 OwnThink.
 *
 * Name        : bilstm_crf.py - 预测
 * Author      : Yener <yener@ownthink.com>
 * Version     : 0.01
 * Description : 
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import viterbi_decode


class Predict(object):
    def __init__(self, model_file):
        with open(model_file, 'rb') as f:
            model, char_to_id, id_to_tag = pickle.load(f)

        self.char_to_id = char_to_id
        self.id_to_tag = {int(k): v for k, v in id_to_tag.items()}
        self.num_class = len(self.id_to_tag)

        graph_def = tf.GraphDef()
        graph_def.ParseFromString(model)

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="prefix")

        self.input_x = graph.get_tensor_by_name("prefix/char_inputs:0")
        self.lengths = graph.get_tensor_by_name("prefix/lengths:0")
        self.dropout = graph.get_tensor_by_name("prefix/dropout:0")
        self.logits = graph.get_tensor_by_name("prefix/project/logits:0")
        self.trans = graph.get_tensor_by_name("prefix/crf_loss/transitions:0")

        self.sess = tf.Session(graph=graph)
        self.sess.as_default()

    def decode(self, logits, trans, sequence_lengths, tag_num):
        small = -1000.0
        viterbi_sequences = []
        start = np.asarray([[small] * tag_num + [0]])
        for logit, length in zip(logits, sequence_lengths):
            score = logit[:length]
            pad = small * np.ones([length, 1])
            score = np.concatenate([score, pad], axis=1)
            score = np.concatenate([start, score], axis=0)
            viterbi_seq, viterbi_score = viterbi_decode(score, trans)
            viterbi_sequences.append(viterbi_seq[1:])
        return viterbi_sequences

    def predict(self, sents):
        inputs = []
        lengths = [len(text) for text in sents]
        max_len = max(lengths)

        for sent in sents:
            sent_ids = [self.char_to_id.get(w) if w in self.char_to_id else self.char_to_id.get("<OOV>") for w in sent]
            padding = [0] * (max_len - len(sent_ids))
            sent_ids += padding
            inputs.append(sent_ids)
        inputs = np.array(inputs, dtype=np.int32)

        feed_dict = {
            self.input_x: inputs,
            self.lengths: lengths,
            self.dropout: 1.0
        }

        logits, trans = self.sess.run([self.logits, self.trans], feed_dict=feed_dict)
        path = self.decode(logits, trans, lengths, self.num_class)
        labels = [[self.id_to_tag.get(l) for l in p] for p in path]
        return labels
