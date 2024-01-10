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

    def __call__(self, rewards):
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
        return np.dot(rewards, self.w)

## Fake squared
# A randomized squared function
class FakeSquared(FakeFunction):
    def __init__(self, dimension=2):
        #self.w = np.zeros(dimension)
        self.randomize()


    # def randomize(self):
    #     w = np.random.random(self.w.shape)
    #     self.w = w / np.sum(w)

    # def calc(self, rewards):
    #     return rewards[:,0] * rewards[:,1]

## Fake logistic
# A randomized squared function
class FakeLogistic(FakeFunction):
    def __init__(self, dimension=2):
        #self.w = np.zeros(dimension)
        self.randomize()


    # def randomize(self):
    #     w = np.random.random(self.w.shape)
    #     self.w = w / np.sum(w)

    # def calc(self, rewards):
    #     return rewards[:,0] * rewards[:,1]

## Fake sin with dimenshing exponenent.
# A randomized squared function
class FakeSinExp(FakeFunction):
    def __init__(self, dimension=2):
        #self.w = np.zeros(dimension)
        self.randomize()


    # def randomize(self):
    #     w = np.random.random(self.w.shape)
    #     self.w = w / np.sum(w)

    # def calc(self, rewards):
    #     return rewards[:,0] * rewards[:,1]



