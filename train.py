# -*- coding: utf-8 -*-
#
# Created by PyCharm
# Date: 16-7-18
# Time: 下午10:31
# Author: Zhu Danxiang
#

import mxnet as mx
import numpy as np
from symbol import make_qacnn_symbol, make_qacnn_inference
from datamanager import QATrainIter, QAEvaluateIter, read_answer, read_vocab
from collections import namedtuple

QAModel = namedtuple('QAModel', ['executor', 'symbol', 'param_block'])
QAInference = namedtuple('QAInference', ['executor'])


def accuracy(pred):
    cnt = np.sum(pred == 0)
    return float(cnt) / len(pred)


def build_model(ctx, symbol, input_shape):
    initializer = mx.initializer.Uniform(scale=0.01)
    arg_shape, _, _ = symbol.infer_shape(**input_shape)
    arg_name = symbol.list_arguments()
    arg_array = [mx.nd.zeros(x, ctx=ctx) for x in arg_shape]

    arg_dict = {k: v for k, v in zip(arg_name, arg_array)}
    grad_dict = {}
    for name, shape in zip(arg_name, arg_shape):
        if name in ['qst', 'pos_ans', 'neg_ans']:
            continue
        grad_dict[name] = mx.nd.zeros(shape, ctx=ctx)

    executor = symbol.bind(ctx=ctx, args=arg_dict, args_grad=grad_dict, grad_req='add')

    param_block = []
    for ix, name in enumerate(arg_name):
        if name in ['qst', 'pos_ans', 'neg_ans']:
            continue
        initializer(name, arg_dict[name])
        param_block.append((ix, arg_dict[name], grad_dict[name], name))

    return QAModel(executor=executor, symbol=symbol, param_block=param_block)


def build_inference(ctx, symbol, arg, input_shape):
    arg_name = symbol.list_arguments()
    arg_dict = {}
    for name, shape in input_shape.iteritems():
        arg_dict[name] = mx.nd.zeros(shape, ctx=ctx)
    for name in arg_name:
        if name in ['qst', 'ans']:
            continue
        arg_dict[name] = arg[name]
    executor = symbol.bind(ctx=ctx, args=arg_dict)
    return QAInference(executor=executor)


def train_step(train_model, train_iter):
    acc = 0.
    loss = 0.
    cnt = 0
    eval_every = 50
    max_grad_norm = 5.0
    optimizer = mx.optimizer.create('rmsprop')
    optimizer.lr = 0.01
    updater = mx.optimizer.get_updater(optimizer)
    train_exec = train_model.executor
    param_block = train_model.param_block
    batch_size = trainiter.batch_size
    for qst, pos_ans, neg_ans in train_iter:
        qst.copyto(train_exec.arg_dict['qst'])
        pos_ans.copyto(train_exec.arg_dict['pos_ans'])
        neg_ans.copyto(train_exec.arg_dict['neg_ans'])
        train_exec.forward(is_train=True)
        train_exec.backward()
        pred = train_exec.outputs[0].asnumpy()
        acc += accuracy(pred)
        loss += np.sum(pred)
        cnt += 1

        norm = 0
        for ix, weight, grad, name in param_block:
            grad /= batch_size
            l2_norm = mx.nd.norm(grad).asscalar()
            norm += l2_norm * l2_norm

        norm = np.sqrt(norm)
        for ix, weight, grad, name in param_block:
            if norm > max_grad_norm:
                grad *= (max_grad_norm / norm)
            updater(ix, grad, weight)
            grad[:] = 0.0
        if cnt % eval_every == 0:
            print 'training   accuracy:', acc / cnt, '\tloss:', loss / cnt


def val_step(inference, val_iter):
    inference_exec = inference.executor
    val_result = {}
    for qid, aid, label, qst, ans in val_iter:
        qst.copyto(inference_exec.arg_dict['qst'])
        ans.copyto(inference_exec.arg_dict['ans'])
        inference_exec.forward(is_train=False)
        pred = inference_exec.outputs[0].asnumpy()
        for i in xrange(len(qid)):
            if qid[i] in val_result:
                val_result[qid[i]].append((pred[i], label[i], aid[i]))
            else:
                val_result[qid[i]] = [(pred[i], label[i], aid[i])]
    cnt = 0
    f = open('data/pred.txt', 'w')
    for qid, v in val_result.iteritems():
        v = sorted(v, key=lambda x: x[0], reverse=True)
        for ix, x in enumerate(v):
            if x[1] == 1:
                pos_ix = ix
                pos_id = x[2]
                break
        best_match = v[0]
        pred, label, aid = best_match[0], best_match[1], best_match[2]
        print >> f, label, pos_ix, '\n%s\n%s\n%s' % (
            'qst: %s' % ' '.join(val_iter.convert_qid_to_groundtruth(qid)),
            'ans: %s' % ' '.join(val_iter.convert_aid_to_groundtruth(pos_id)),
            'pos: %s' % ' '.join(val_iter.convert_aid_to_groundtruth(aid)))
        if label == 1:
            cnt += 1
    f = open('data/test_acc.txt', 'a')
    print >> f, '\tevaluate   accuracy:', float(cnt) / len(
        val_result), ' pos cnt:', cnt, ' neg cnt:', len(val_result) - cnt
    f.close()


if __name__ == '__main__':
    base_path = '/home/pig/Data/insuranceQA/V1/'
    train_file = base_path + 'question.train.token_idx.label'
    val_file = base_path + 'question.test1.label.token_idx.pool'
    vocab_file = base_path + 'vocabulary'
    answer_file = base_path + 'answers.label.token_idx'
    margin = 0.05
    filter_sizes = [2, 3, 4]
    num_filter = 200
    max_length = 200
    embedding_size = 200
    dropout = 0
    batch_size = 10
    epoch = 50000
    ctx = mx.gpu()
    vocab = read_vocab(vocab_file)
    vocab_size = len(vocab) + 2
    print 'vocab size:', vocab_size
    idx2answer = read_answer(answer_file)
    trainiter = QATrainIter(train_file, vocab, idx2answer, batch_size, max_length)
    print 'train size:', trainiter.size
    valiter = QAEvaluateIter(val_file, vocab, idx2answer, batch_size, max_length)
    print 'val size:', valiter.size

    symbol = make_qacnn_symbol(margin=margin, filter_sizes=filter_sizes, num_filter=num_filter, seqlen=max_length,
                               vocab_size=vocab_size, embedding_size=embedding_size, dropout=dropout)
    inference_symbol = make_qacnn_inference(filter_sizes=filter_sizes, num_filter=num_filter, seqlen=max_length,
                                            vocab_size=vocab_size, embedding_size=embedding_size, dropout=0.)
    model = build_model(ctx, symbol, dict(trainiter.provide_data))
    arg = {v[3]: v[1] for v in model.param_block}
    inference = build_inference(ctx, inference_symbol, arg, dict(valiter.provide_data))

    for i in xrange(epoch):
        train_step(model, trainiter)
        trainiter.reset()
        if i >= 20 and i % 5 == 0:
            val_step(inference, valiter)
            valiter.reset()
