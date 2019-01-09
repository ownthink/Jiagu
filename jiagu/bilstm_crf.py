import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import viterbi_decode


class Predict(object):
    def __init__(self, model_file, char_to_id={}, id_to_tag={}, use_pack=True):
        if use_pack:
            with open(model_file, 'rb') as f:
                model, char_to_id, id_to_tag = pickle.load(f)
        else:
            model = model_file

        self.char_to_id = char_to_id
        self.id_to_tag = {int(k): v for k, v in id_to_tag.items()}
        self.graph = self.load_graph(model, use_pack)

        self.input_x = self.graph.get_tensor_by_name("prefix/char_inputs:0")
        self.lengths = self.graph.get_tensor_by_name("prefix/lengths:0")
        self.dropout = self.graph.get_tensor_by_name("prefix/dropout:0")
        self.logits = self.graph.get_tensor_by_name("prefix/project/logits:0")
        self.trans = self.graph.get_tensor_by_name("prefix/crf_loss/transitions:0")

        self.sess = tf.Session(graph=self.graph)
        self.sess.as_default()
        self.num_class = len(self.id_to_tag)

    def load_graph(self, model_file, use_pack):
        if use_pack:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(model_file)
        else:
            with tf.gfile.GFile(model_file, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="prefix")
        return graph

    def decode(self, logits, trans, sequence_lengths, tag_num):
        viterbi_sequences = []
        small = -1000.0
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
