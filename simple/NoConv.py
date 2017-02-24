import io
import os
import sys
import argparse
import glob
from PIL import Image
import random
import pickle
import collections
import bz2
import itertools

import numpy as np

def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

def loadDataset():
    sources = []
    targets = []
    l = []
    for x in xrange(1, 15):
        l.append(x)
    random.shuffle(l)

    for x in l:
        #.transpose(2,0,1)
        targets.append(np.asarray(Image.open('./images/tiles_{0}.png'.format(x))).flatten()/255.)
        sources.append(np.asarray(Image.open('./images/smalltiles_{0}.png'.format(x))).transpose(2,0,1).flatten()/255.)

    return (np.asarray(sources), np.asarray(targets))

def main():
    sources, targets = loadDataset()
    np.random.seed(1)
    weights = 2*np.random.random((12,48)) - 1 # initialize 768 random weights

    for x in xrange(0, 14): # training iterations
        currentInput = sources[x]
        expectedOutput = targets[x]

        print 'Dotting {0} with {1}.'.format(currentInput.shape, weights.shape)
        output = nonlin(np.dot(currentInput,weights))
        error = expectedOutput - output

        if x % 100 == 0:
            print error.mean()

        # calculate derivitive to update weights appropriatly
        delta = error * nonlin(output,True)

        # update weights
        
        currentInput = np.expand_dims(currentInput, axis=1)
        delta = np.expand_dims(delta, axis=0)

        weights += np.dot(currentInput,delta)

    np.save('weights.npy', weights)
    predictedOut = (nonlin(np.dot(sources[0],weights))*255).reshape(3, 4, 4).astype('uint8').transpose(1,2,0)
    Image.fromarray(predictedOut).save("OUT.png")

if __name__ == '__main__':
    main()