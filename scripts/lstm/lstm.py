#!/usr/bin/env python

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import chainer
import chainer.functions as F
import chainer.initializers as I
import chainer.links as L

class LSTM(chainer.Chain):
    def __init__(self, n_class=1000):
        super(LSTM, self).__init__()
        insize = 154587
        #insize = 3 * 227
        hidden_size = 64
        self.n_class = n_class
        with self.init_scope():
            self.l1 = L.LSTM(insize, hidden_size)
            self.l2 = L.Linear(hidden_size, n_class)

    def forward(self, x, t=None):
        h = self.l1(x)
        #print(h.shape)
        h = self.l2(h)
        #print(h.shape)

        self.pred = F.softmax(h)
        if t is None:
            assert not chainer.config.train
            return

        self.loss = F.softmax_cross_entropy(h, t)
        self.acc = F.accuracy(self.pred, t)

        chainer.report({"loss": self.loss, "accuracy": self.acc}, self)

        return self.loss

class LSTM_2(chainer.Chain):
    def __init__(self, n_class=1):
        super(LSTM_2, self).__init__()
        insize = 154587
        hidden_size = 64
        self.n_class = n_class
        with self.init_scope():
            self.l1 = L.LSTM(insize, hidden_size)
            self.l2 = L.Linear(hidden_size, n_class)

    def forward(self, x, t=None):
        #print(x.shape)
        h = self.l1(x)
        #print(h.shape)
        h = self.l2(h)

        #print(h.shape)

        #self.pred = F.sigmoid(h)
        self.pred = h
        #print(self.pred.data.shape)
        if t is None:
            assert not chainer.config.train
            return

        #self.loss = F.sigmoid_cross_entropy(h, t)

        if t is not None:
            t = t.reshape(-1,1)
        self.loss = F.mean_squared_error(h, t)

        chainer.report({"loss": self.loss}, self)
        return self.loss

class LSTM_torch(nn.Module):
    def __init__(self,  n_class=2):
        super(LSTM_torch, self).__init__()
        self.seq_len = 227
        self.feature_size = 227 * 3
        self.hidden_layer_size = 128
        self.rnn_layers = 1

        self.n_class = n_class
        #print(self.n_class)
        self.simple_rnn = nn.LSTM(input_size = self.feature_size,
                                 hidden_size = self.hidden_layer_size,
                                 num_layers = self.rnn_layers)
        self.fc = nn.Linear(self.hidden_layer_size, self.n_class)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.rnn_layers, batch_size, self.hidden_layer_size)
        cell = torch.zeros(self.rnn_layers, batch_size, self.hidden_layer_size)
        return (hidden, cell)
    
    def forward(self, x, h=None):
        #print(x.shape)
        batch_size = x.shape[0]
        self.hidden_cell = self.init_hidden(batch_size)
        self.hidden = self.hidden_cell[0].to("cuda")
        self.cell = self.hidden_cell[1].to("cuda")
        x = x.view(batch_size, self.feature_size, self.seq_len) #(batch, channel, height, width) > (batch, height(3channel), width) = (batch, feature, sequence)
        x = x.permute(2,0,1)
        #print(x.shape)

        output, (h_n, c_n) = self.simple_rnn(x, (self.hidden, self.cell)) # RNN input - (seq, batch, feature)
        x = h_n[-1,:,:]
        x = self.fc(x)

        #self.pred = F.softmax(x)
        return x
