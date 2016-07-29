# -*- coding: utf-8 -*-
#
# Created by PyCharm
# Date: 16-7-18
# Time: 下午9:01
# Author: Zhu Danxiang
#

import mxnet as mx
import numpy as np


def read_vocab(filename):
    vocab = {}
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            items = line.split()
            if items[0] not in vocab:
                vocab[items[0]] = items[1]
    return vocab


def get_word2ix(vocab):
    word2ix = {'<PAD>': 0, '<UNK>': 1}
    code = len(word2ix)
    for word in vocab.keys():
        if word not in word2ix:
            word2ix[word] = code
            code += 1
    return word2ix


def read_answer(filename):
    idx2answer = {}
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            idx, answer = line.split('\t')
            idx2answer[int(idx)] = answer.split(' ')
    return idx2answer


def pad_sentence(sent, pad_symbol, length):
    if len(sent) < length:
        sent = sent + [pad_symbol] * (length - len(sent))
    return sent


def encode_sentence(sent, vocab, max_length=None):
    encoded = []
    for word in sent:
        if word == '':
            continue
        if word in vocab:
            encoded.append(vocab[word])
        else:
            encoded.append(vocab['<UNK>'])
    if max_length is None:
        return encoded
    else:
        encoded = pad_sentence(encoded, vocab['<PAD>'], max_length)
        return encoded[:max_length]


def build_dataset(filename):
    dataset = []
    idx2question = {}
    idx = 0
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            items = line.split('\t')
            idx2question[idx] = items[0].split(' ')
            qst = idx
            ans_pool = items[1].split(' ')
            dataset.append((qst, ans_pool))
            idx += 1
    return dataset, idx2question


class QATrainIter(mx.io.DataIter):
    def __init__(self, filename, vocab, idx2answer, batch_size, max_length):
        self.vocab = vocab
        self.word2ix = get_word2ix(vocab)
        self.idx2answer = idx2answer
        self.batch_size = batch_size
        self.max_length = max_length
        self.dataset, self.idx2question = build_dataset(filename)

        self.provide_data = [('qst', (self.batch_size, self.max_length)),
                             ('pos_ans', (self.batch_size, self.max_length)),
                             ('neg_ans', (self.batch_size, self.max_length))]
        self.reset()

    def __iter__(self):
        for ix in xrange(self.size / self.batch_size):
            qst = []
            pos_ans = []
            neg_ans = []
            for i in xrange(self.batch_size):
                rand_ix = np.random.randint(0, self.size - 1)
                qst.append(encode_sentence(self.idx2question[self.qst[rand_ix]], self.word2ix, self.max_length))
                pos_ans.append(encode_sentence(self.idx2answer[self.pos_ans[rand_ix]], self.word2ix, self.max_length))
                neg_ans.append(encode_sentence(self.idx2answer[self.neg_ans[rand_ix]], self.word2ix, self.max_length))
            qst = mx.nd.array(qst)
            pos_ans = mx.nd.array(pos_ans)
            neg_ans = mx.nd.array(neg_ans)
            yield qst, pos_ans, neg_ans

    def reset(self):
        self.qst = []
        self.pos_ans = []
        self.neg_ans = []
        for data in self.dataset:
            qid, ans_pool = data[0], data[1]
            for aid in ans_pool:
                rand_ix = np.random.randint(1, len(self.idx2answer))
                self.qst.append(int(qid))
                self.pos_ans.append(int(aid))
                self.neg_ans.append(rand_ix)
        self.size = len(self.qst)


def build_testset(filename):
    dataset = []
    idx2question = {}
    idx = 0
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            items = line.split('\t')
            idx2question[idx] = items[1].split(' ')
            qst = idx
            pos_ans_pool = items[0].split(' ')
            ans_pool = items[2].split(' ')
            dataset.append((qst, pos_ans_pool, ans_pool))
            idx += 1
    return dataset, idx2question


class QAEvaluateIter(mx.io.DataIter):
    def __init__(self, filename, vocab, idx2answer, batch_size, max_length):
        self.vocab = vocab
        self.word2ix = get_word2ix(vocab)
        self.idx2answer = idx2answer
        self.batch_size = batch_size
        self.max_length = max_length
        self.dataset, self.idx2question = build_testset(filename)
        self.label = []
        self.qid = []
        self.aid = []

        for data in self.dataset:
            qid, pos_ans_pool, ans_pool = data[0], data[1], data[2]
            for aid in ans_pool:
                self.qid.append(int(qid))
                self.aid.append(int(aid))
                if aid in pos_ans_pool:
                    self.label.append(1)
                else:
                    self.label.append(0)

        self.label = np.array(self.label)
        self.qid = np.array(self.qid)
        self.aid = np.array(self.aid)

        self.size = len(self.qid)

        self.provide_data = [('qst', (self.batch_size, self.max_length)),
                             ('ans', (self.batch_size, self.max_length))]
        self.reset()

    def __iter__(self):
        for ix in xrange(0, self.size, self.batch_size):
            if ix + self.batch_size > self.size:
                continue
            qid = self.qid[ix: ix + self.batch_size]
            aid = self.aid[ix: ix + self.batch_size]
            label = self.label[ix: ix + self.batch_size]
            qst = mx.nd.array([encode_sentence(self.idx2question[x], self.word2ix, self.max_length) for x in qid])
            ans = mx.nd.array([encode_sentence(self.idx2answer[x], self.word2ix, self.max_length) for x in aid])
            yield qid, aid, label, qst, ans

    def reset(self):
        rand_ix = np.random.permutation(len(self.qid))
        self.label = self.label[rand_ix]
        self.qid = self.qid[rand_ix]
        self.aid = self.aid[rand_ix]

    def convert_qid_to_groundtruth(self, qid):
        return encode_sentence(self.idx2question[qid], self.vocab)

    def convert_aid_to_groundtruth(self, aid):
        return encode_sentence(self.idx2answer[aid], self.vocab)
