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

def parseArgs():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', default='./dataset/')
    parser.add_argument('--weights', default=None)
    parser.add_argument('--batchSize', '-b', default=5)
    parser.add_argument('--weightSaveInterval', '-s', default=10000000)
    parser.add_argument('--printError', '-e', default=10000)
    parser.add_argument('--learningRate', '-lr', default=1)
    parser.add_argument('--learningRateDecay', '-lrd', default=10.0)
    parser.add_argument('--learningRateInterval', '-lri', default=1000000)
    parser.add_argument('--trainingEpochs', '-te', default=50)
    parser.add_argument('--metricInterval', default=1000)
    parser.add_argument('--metricsFile', default='metrics.csv')
    parser.add_argument('--metrics', default=False, action='store_true')

    args = parser.parse_args()

    return args

def loadDataset(args):
    sources = []
    targets = []
    l = []

    # Generate a shuffled list of ints from 1 to
    # the number of samples. 
    allImages = glob.glob(args.dataset + '*')
    for x in xrange(1, len(allImages) / 2):
        l.append(x)
    random.shuffle(l)

    print 'Loading {0} images...'.format(len(l)+1) 

    # For each sample, convert to RGB, rotate dimensions, flatten into a 1d array, and divide by 255
    # so each value is between 0 and 1.
    for x in l:
        targets.append(np.asarray(Image.open(args.dataset + 'big-{0}.png'.format(x)).convert('RGB')).transpose(2,0,1).flatten()/255.)
        sources.append(np.asarray(Image.open(args.dataset + 'small-{0}.png'.format(x)).convert('RGB')).transpose(2,0,1).flatten()/255.)

    return (np.asarray(sources), np.asarray(targets))

def main():
    args = parseArgs()
    sources, targets = loadDataset(args)

    sourceSize = sources[0].shape[0]
    targetSize = targets[0].shape[0]
    source = sources[0].shape
    target = targets[0].shape

    # Load weights if arg provided, otherwise generate random ones.
    if args.weights != None:
        weights = np.load(args.weights)
    else:
        weights = 2*np.random.random((sourceSize,targetSize)) - 1

    # Clear previous metrics file
    if args.metrics:
        with open(args.metricsFile, 'w+') as f:
            f.write('epoch,error,learningRate\n')

    learningRate = args.learningRate
    batchedWeightUpdate = np.zeros((sourceSize,targetSize))
    totalError = 0
    for epoch in xrange(0, args.trainingEpochs):
        epochError = 0
        for x in xrange(0, len(sources)):
            currentInter = ((len(sources) * epoch) + x)
            currentInput = sources[x]
            expectedOutput = targets[x]

            if currentInput.shape != source:
                continue

            if expectedOutput.shape != target:
                continue

            # Predict an output from currentInput
            # Calculate the difference with the actual image
            output = nonlin(np.dot(currentInput,weights))
            error = expectedOutput - output
            totalError += abs(error)
            epochError += abs(error)

            if currentInter % args.printError == 0:
                print 'Total Error: ' + str(totalError.mean() / currentInter)

            # Calculate derivitive to understand which weights contributed what
            delta = error * nonlin(output,True) * learningRate

            # Calculate weight update
            currentInput = np.expand_dims(currentInput, axis=1)
            delta = np.expand_dims(delta, axis=0)

            # Update weights by average batches
            batchedWeightUpdate += np.dot(currentInput,delta)
            if currentInter % args.batchSize == 0:
                weights += (batchedWeightUpdate / args.batchSize)
                batchedWeightUpdate = np.zeros((sourceSize,targetSize))

            # Occasionally save weights
            if currentInter % args.weightSaveInterval == 0:
                np.save('weights_{0}.npy'.format(currentInter), weights)

            # Decay learning rate over time
            if currentInter % args.learningRateInterval == 0:
                learningRate = learningRate / (args.learningRateDecay * 1.0)
                print 'Learning rate: ' + str(learningRate)

            # Output metrics
            if args.metrics and currentInter % int(args.metricInterval) == 0:
                with open(args.metricsFile, 'a+') as f:
                    f.write('{0},{1},{2}\n'.format(currentInter, totalError.mean() / currentInter, learningRate))

        print 'Epoch Error: ' + str(epochError.mean() / len(sources))

    np.save('weights_{0}.npy'.format(args.trainingEpochs * len(sources)), weights)

if __name__ == '__main__':
    main()