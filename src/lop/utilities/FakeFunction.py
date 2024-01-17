# Copyright 2024 Ian Rankin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
# to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# FakeFunction.py
# Written Ian Rankin - January 2024
#
# A set of fake user functions for testing the learning models.
# Includes randomization for full tests. Some of this code I wrote in 2022.

import numpy as np

## Fake function
# Abstract class for a Fake Function for experiments with different learning
# methods for synthetic users.
class FakeFunction:
    def calc(self, rewards):
        raise NotImplementedError('FakeFunction calc is not implemented')

    def randomize(self):
        raise NotImplementedError('FakeFunction randomize is not implemented')

    def __call__(self, rewards, data=None):
        return self.calc(rewards)

## Fake linear
# A reasonably simple linear function to test with.
class FakeLinear(FakeFunction):
    def __init__(self, dimension=2):
        self.w = np.zeros(dimension)
        self.randomize()


    def randomize(self):
        w = np.random.random(self.w.shape)
        self.w = w / np.sum(w)

    def calc(self, rewards):
        if self.w.shape[0] == 1:
            if isinstance(rewards, list):
                rewards = np.array(rewards)
            if isinstance(rewards, np.ndarray):
                return rewards * self.w
            else:
                return (rewards * self.w)[0]
        return np.dot(rewards, self.w)

## Fake squared
# A randomized squared function
class FakeSquared(FakeFunction):
    def __init__(self, dimension=2):
        self.w = np.zeros(dimension)
        self.randomize()


    def randomize(self):
        w = np.random.random(self.w.shape)
        self.w = w / np.sum(w)

    def calc(self, rewards):
        if self.w.shape[0] == 1:
            if isinstance(rewards, list):
                rewards = np.array(rewards)
            if isinstance(rewards, np.ndarray):
                return rewards*rewards * self.w[0]
            else:
                return (rewards*rewards * self.w[0])

        return np.dot(rewards*rewards, self.w)

## Fake logistic
# A randomized logistic function
class FakeLogistic(FakeFunction):
    def __init__(self, dimension=2):
        self.w = np.zeros(dimension)
        self.randomize()


    def randomize(self):
        w = np.random.random(self.w.shape) * 2
        self.w = w

        self.A = 0
        self.K = 1.0
        self.C = 1.0
        self.Q = (np.random.random()*3)**2
        self.v = np.random.random()*2

    def calc(self, rewards):
        if self.w.shape[0] == 1:
            if isinstance(rewards, list):
                rewards = np.array(rewards)
            wx = rewards * self.w[0]
        else:
            wx = np.dot(rewards, self.w)
        
        logi = self.A + ((self.K - self.A) / (self.C + self.Q*np.exp(-wx)**(1/self.v)))
        return logi

## Fake sin with dimenshing exponenent.
#
class FakeSinExp(FakeFunction):
    def __init__(self, dimension=2):
        self.w = np.zeros(dimension)
        self.randomize()


    def randomize(self):
        w = np.random.random(self.w.shape)
        self.w = w

        k = np.random.random(self.w.shape)
        self.k = k / np.sum(k)

        self.phase = np.random.random()*np.pi

    def calc(self, rewards):
        if self.w.shape[0] == 1:
            if isinstance(rewards, list):
                rewards = np.array(rewards)
            wr = rewards * self.w[0]
            kr = rewards * self.k[0]
        else:
            kr = np.dot(rewards, self.k)
            wr = np.dot(rewards, self.w)

        return np.sin(kr+self.phase) * np.exp(-wr)



