#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy, sys
import numpy as np
from collections import deque

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Chain, optimizers, Variable, serializers

class Neuralnet(Chain):

    def __init__(self, n_in, n_out):
        super(Neuralnet, self).__init__(
            L1 = L.Linear(n_in, 100),
            L2 = L.Linear(100, 100),
            L3 = L.Linear(100, 100),
            Q_value = L.Linear(100, n_out, initialW=np.zeros((n_out, 100), dtype=np.float32))
        )

    def Q_func(self, x):
        h = F.leaky_relu(self.L1(x))
        h = F.leaky_relu(self.L2(h))
        h = F.leaky_relu(self.L3(h))
        h = self.Q_value(h)
        return h

class Agent():

    def __init__(self, n_st, n_act, seed):
        np.random.seed(seed)
        sys.setrecursionlimit(10000)
        self.n_act = n_act
        self.model = Neuralnet(n_st, n_act)
        self.target_model = copy.deepcopy(self.model)
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)
        self.memory = deque()
        self.loss = 0
        self.step = 0
        self.gamma = 0.99
        self.mem_size = 1000
        self.batch_size = 100
        self.epsilon = 1
        self.epsilon_decay = 0.005
        self.epsilon_min = 0
        self.exploration = 1000
        self.train_freq = 10
        self.target_update_freq = 20

    def stock_experience(self, st, act, r, st_dash, ep_end):
        self.memory.append((st, act, r, st_dash, ep_end))
        if len(self.memory) > self.mem_size:
            self.memory.popleft()

    def forward(self, st, act, r, st_dash, ep_end):
        s = Variable(st)
        s_dash = Variable(st_dash)
        Q = self.model.Q_func(s)
        tmp = self.target_model.Q_func(s_dash)
        tmp = list(map(np.max, tmp.data))
        max_Q_dash = np.asanyarray(tmp, dtype=np.float32)
        target = np.asanyarray(copy.deepcopy(Q.data), dtype=np.float32)
        for i in xrange(self.batch_size):
            target[i, act[i]] = r[i] + (self.gamma * max_Q_dash[i]) * (not ep_end[i])
        loss = F.mean_squared_error(Q, Variable(target))
        self.loss = loss.data
        return loss

    def suffle_memory(self):
        mem = np.array(self.memory)
        return np.random.permutation(mem)

    def parse_batch(self, batch):
        st, act, r, st_dash, ep_end = [], [], [], [], []
        for i in xrange(self.batch_size):
            st.append(batch[i][0])
            act.append(batch[i][1])
            r.append(batch[i][2])
            st_dash.append(batch[i][3])
            ep_end.append(batch[i][4])
        st = np.array(st, dtype=np.float32)
        act = np.array(act, dtype=np.int8)
        r = np.array(r, dtype=np.float32)
        st_dash = np.array(st_dash, dtype=np.float32)
        ep_end = np.array(ep_end, dtype=np.bool)
        return st, act, r, st_dash, ep_end

    def experience_replay(self):
        mem = self.suffle_memory()
        perm = np.array(xrange(len(mem)))
        for start in perm[::self.batch_size]:
            index = perm[start:start+self.batch_size]
            batch = mem[index]
            st, act, r, st_d, ep_end = self.parse_batch(batch)
            self.model.zerograds()
            loss = self.forward(st, act, r, st_d, ep_end)
            loss.backward()
            self.optimizer.update()

    def get_action(self, st):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.n_act), 0
        else:
            s = Variable(st)
            Q = self.model.Q_func(s)
            Q = Q.data[0]
            a = np.argmax(Q)
            return np.asarray(a, dtype=np.int8), max(Q)

    def reduce_epsilon(self):
        if self.epsilon > self.epsilon_min and self.exploration < self.step:
            self.epsilon -= self.epsilon_decay

    def train(self):
        if len(self.memory) >= self.mem_size:
            if self.step % self.train_freq == 0:
                self.experience_replay()
                self.reduce_epsilon()
            if self.step % self.target_update_freq == 0:
                self.target_model = copy.deepcopy(self.model)
        self.step += 1

    def save_model(self, model_dir):
        serializers.save_npz(model_dir + "model.npz", self.model)

    def load_model(self, model_dir):
        serializers.load_npz(model_dir + "model.npz", self.model)
        self.target_model = copy.deepcopy(self.model)
