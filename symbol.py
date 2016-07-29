# -*- coding: utf-8 -*-
#
# Created by PyCharm
# Date: 16-7-18
# Time: 下午9:56
# Author: Zhu Danxiang
#

import mxnet as mx


def make_qacnn_symbol(margin, filter_sizes, num_filter, seqlen, vocab_size, embedding_size, dropout):
    question = mx.sym.Variable('qst')
    pos_ans = mx.sym.Variable('pos_ans')
    neg_ans = mx.sym.Variable('neg_ans')
    embed_weight = mx.sym.Variable('embed_weight')
    qst_emb = mx.sym.Embedding(data=question, input_dim=vocab_size, output_dim=embedding_size, weight=embed_weight)
    pos_emb = mx.sym.Embedding(data=pos_ans, input_dim=vocab_size, output_dim=embedding_size, weight=embed_weight)
    neg_emb = mx.sym.Embedding(data=neg_ans, input_dim=vocab_size, output_dim=embedding_size, weight=embed_weight)

    qst_emb_reshape = mx.sym.Reshape(data=qst_emb, shape=(-1, 1, seqlen, embedding_size))
    pos_emb_reshape = mx.sym.Reshape(data=pos_emb, shape=(-1, 1, seqlen, embedding_size))
    neg_emb_reshape = mx.sym.Reshape(data=neg_emb, shape=(-1, 1, seqlen, embedding_size))

    qst_outs = []
    pos_outs = []
    neg_outs = []
    for filter_size in filter_sizes:
        conv_weight = mx.sym.Variable('conv_%d_weight' % filter_size)
        conv_bias = mx.sym.Variable('conv_%d_bias' % filter_size)
        qst_conv = mx.sym.Convolution(data=qst_emb_reshape, kernel=(filter_size, embedding_size), num_filter=num_filter,
                                      weight=conv_weight, bias=conv_bias)
        qst_active = mx.sym.Activation(data=qst_conv, act_type='relu')
        qst_pool = mx.sym.Pooling(data=qst_active, pool_type='max', kernel=(seqlen - filter_size + 1, 1))
        qst_outs.append(qst_pool)
        # qst_pool = mx.sym.Pooling(data=qst_conv, pool_type='max', kernel=(seqlen - filter_size + 1, 1))
        # qst_active = mx.sym.Activation(data=qst_pool, act_type='tanh')
        # qst_outs.append(qst_active)

        pos_conv = mx.sym.Convolution(data=pos_emb_reshape, kernel=(filter_size, embedding_size), num_filter=num_filter,
                                      weight=conv_weight, bias=conv_bias)
        pos_active = mx.sym.Activation(data=pos_conv, act_type='relu')
        pos_pool = mx.sym.Pooling(data=pos_active, pool_type='max', kernel=(seqlen - filter_size + 1, 1))
        pos_outs.append(pos_pool)
        # pos_pool = mx.sym.Pooling(data=pos_conv, pool_type='max', kernel=(seqlen - filter_size + 1, 1))
        # pos_active = mx.sym.Activation(data=pos_pool, act_type='tanh')
        # pos_outs.append(pos_active)

        neg_conv = mx.sym.Convolution(data=neg_emb_reshape, kernel=(filter_size, embedding_size), num_filter=num_filter,
                                      weight=conv_weight, bias=conv_bias)
        neg_active = mx.sym.Activation(neg_conv, act_type='relu')
        neg_pool = mx.sym.Pooling(data=neg_active, pool_type='max', kernel=(seqlen - filter_size + 1, 1))
        neg_outs.append(neg_pool)
        # neg_pool = mx.sym.Pooling(data=neg_conv, pool_type='max', kernel=(seqlen - filter_size + 1, 1))
        # neg_active = mx.sym.Activation(data=neg_pool, act_type='tanh')
        # neg_outs.append(neg_active)

    total_filter = num_filter * len(filter_sizes)
    qst_concat = mx.sym.Reshape(data=mx.symbol.Concat(*qst_outs, dim=1), shape=(-1, total_filter))
    pos_concat = mx.sym.Reshape(data=mx.symbol.Concat(*pos_outs, dim=1), shape=(-1, total_filter))
    neg_concat = mx.sym.Reshape(data=mx.symbol.Concat(*neg_outs, dim=1), shape=(-1, total_filter))

    if dropout > 0:
        qst_concat = mx.sym.Dropout(data=qst_concat, p=dropout)
        pos_concat = mx.sym.Dropout(data=pos_concat, p=dropout)
        neg_concat = mx.sym.Dropout(data=neg_concat, p=dropout)

    qst_norm = mx.sym.sqrt(data=mx.sym.sum(data=qst_concat * qst_concat, axis=1))
    pos_norm = mx.sym.sqrt(data=mx.sym.sum(data=pos_concat * pos_concat, axis=1))
    neg_norm = mx.sym.sqrt(data=mx.sym.sum(data=neg_concat * neg_concat, axis=1))

    pos_numer = mx.sym.sum(data=qst_concat * pos_concat, axis=1)
    neg_numer = mx.sym.sum(data=qst_concat * neg_concat, axis=1)

    pos_denom = qst_norm * pos_norm
    neg_denom = qst_norm * neg_norm

    pos_cosin = pos_numer / pos_denom
    neg_cosin = neg_numer / neg_denom

    loss = mx.sym.maximum(0, margin - (pos_cosin - neg_cosin))
    return mx.sym.MakeLoss(data=loss, name='loss')


def make_qacnn_inference(filter_sizes, num_filter, seqlen, vocab_size, embedding_size, dropout):
    question = mx.sym.Variable('qst')
    ans = mx.sym.Variable('ans')
    embed_weight = mx.sym.Variable('embed_weight')

    qst_emb = mx.sym.Embedding(data=question, input_dim=vocab_size, output_dim=embedding_size, weight=embed_weight)
    ans_emb = mx.sym.Embedding(data=ans, input_dim=vocab_size, output_dim=embedding_size, weight=embed_weight)

    qst_emb_reshape = mx.sym.Reshape(data=qst_emb, shape=(-1, 1, seqlen, embedding_size))
    ans_emb_reshape = mx.sym.Reshape(data=ans_emb, shape=(-1, 1, seqlen, embedding_size))

    qst_outs = []
    ans_outs = []
    for filter_size in filter_sizes:
        conv_weight = mx.sym.Variable('conv_%d_weight' % filter_size)
        conv_bias = mx.sym.Variable('conv_%d_bias' % filter_size)
        qst_conv = mx.sym.Convolution(data=qst_emb_reshape, kernel=(filter_size, embedding_size), num_filter=num_filter,
                                      weight=conv_weight, bias=conv_bias)
        qst_active = mx.sym.Activation(data=qst_conv, act_type='relu')
        qst_pool = mx.sym.Pooling(data=qst_active, pool_type='max', kernel=(seqlen - filter_size + 1, 1))
        qst_outs.append(qst_pool)
        # qst_pool = mx.sym.Pooling(data=qst_conv, pool_type='max', kernel=(seqlen - filter_size + 1, 1))
        # qst_active = mx.sym.Activation(data=qst_pool, act_type='tanh')
        # qst_outs.append(qst_active)

        ans_conv = mx.sym.Convolution(data=ans_emb_reshape, kernel=(filter_size, embedding_size), num_filter=num_filter,
                                      weight=conv_weight, bias=conv_bias)
        ans_active = mx.sym.Activation(data=ans_conv, act_type='relu')
        ans_pool = mx.sym.Pooling(data=ans_active, pool_type='max', kernel=(seqlen - filter_size + 1, 1))
        ans_outs.append(ans_pool)
        # ans_pool = mx.sym.Pooling(data=ans_conv, pool_type='max', kernel=(seqlen - filter_size + 1, 1))
        # ans_active = mx.sym.Activation(data=ans_pool, act_type='tanh')
        # ans_outs.append(ans_active)

    total_filter = num_filter * len(filter_sizes)
    qst_concat = mx.sym.Reshape(data=mx.symbol.Concat(*qst_outs, dim=1), shape=(-1, total_filter))
    ans_concat = mx.sym.Reshape(data=mx.symbol.Concat(*ans_outs, dim=1), shape=(-1, total_filter))

    if dropout > 0:
        qst_concat = mx.sym.Dropout(data=qst_concat, p=dropout)
        ans_concat = mx.sym.Dropout(data=ans_concat, p=dropout)

    qst_norm = mx.sym.sqrt(data=mx.sym.sum(data=qst_concat * qst_concat, axis=1))
    ans_norm = mx.sym.sqrt(data=mx.sym.sum(data=ans_concat * ans_concat, axis=1))
    ans_cosin = mx.sym.sum(data=qst_concat * ans_concat, axis=1) / (qst_norm * ans_norm)

    return ans_cosin
