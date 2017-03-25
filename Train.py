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
    parser.add_argument('--trainingEpochs', '-te', default=20)
    parser.add_argument('--metricInterval', default=1000)
    parser.add_argument('--metricsFile', default='metrics.csv')
    parser.add_argument('--metrics', default=False, action='store_true')
    parser.add_argument('--prefix', default='')

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

    print 'Loading {0} training samples...'.format(len(l)+1) 

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
        firstLayer = weights[0]
        secondLayer = weights[1]
    else:
        firstLayer = 2*np.random.random((sourceSize,targetSize)) - 1
        secondLayer = 2*np.random.random((targetSize,targetSize)) - 1

    # Clear previous metrics file
    if args.metrics:
        with open(args.metricsFile, 'w+') as f:
            f.write('epoch,error,learningRate\n')

    learningRate = args.learningRate
    batchedWeightUpdate1 = np.zeros((sourceSize,targetSize))
    batchedWeightUpdate2 = np.zeros((targetSize,targetSize))
    totalError = 0
    for epoch in xrange(0, args.trainingEpochs):
        epochError = 0
        for x in xrange(1, len(sources)):
            currentInter = ((len(sources) * epoch) + x)
            currentInput = sources[x]
            expectedOutput = targets[x]

            # These continues are for weird edge cases where imagemagick does not
            # create the right sized image.
            if currentInput.shape != source:
                continue

            if expectedOutput.shape != target:
                continue

            # Predict an output from currentInput
            # Calculate the difference with the actual image
            output1 = nonlin(np.dot(currentInput,firstLayer))
            output2 = nonlin(np.dot(output1,secondLayer))
            error2 = expectedOutput - output2
            totalError += abs(error2)
            epochError += abs(error2)

            if currentInter % args.printError == 0:
                print 'Total Error: ' + str(totalError.mean() / currentInter)

            # Calculate derivitive to understand which weights contributed what
            delta2 = error2 * nonlin(output2,deriv=True) * learningRate

            error1 = delta2.dot(secondLayer.T)

            delta1 = error1 * nonlin(output1,deriv=True)

            # Calculate weight update
            currentInput = np.expand_dims(currentInput, axis=1)
            delta2 = np.expand_dims(delta2, axis=0)

            output1 = np.expand_dims(output1, axis=1)
            delta1 = np.expand_dims(delta1, axis=0)
            
            # Update weights by average batches
            batchedWeightUpdate2 += output1.dot(delta2)
            if currentInter % args.batchSize == 0:
                secondLayer += (batchedWeightUpdate2 / args.batchSize)
                batchedWeightUpdate2 = np.zeros((targetSize,targetSize))
            
            batchedWeightUpdate1 += currentInput.dot(delta1)
            if currentInter % args.batchSize == 0:
                firstLayer += (batchedWeightUpdate1 / args.batchSize)
                batchedWeightUpdate1 = np.zeros((sourceSize,targetSize))

            # Occasionally save weights
            if currentInter % args.weightSaveInterval == 0:
                np.save(args.prefix + 'weights_{0}.npy'.format(currentInter), np.asarray([firstLayer, secondLayer]))

            # Decay learning rate over time
            if currentInter % args.learningRateInterval == 0:
                learningRate = learningRate / (args.learningRateDecay * 1.0)
                print 'Learning rate: ' + str(learningRate)

            # Output metrics
            if args.metrics and currentInter % int(args.metricInterval) == 0:
                with open(args.metricsFile, 'a+') as f:
                    f.write('{0},{1},{2}\n'.format(currentInter, totalError.mean() / currentInter, learningRate))

        #print 'Epoch Error: ' + str(epochError.mean() / len(sources))

    np.save(args.prefix + 'weights_{0}.npy'.format(args.trainingEpochs * len(sources)), np.asarray([firstLayer, secondLayer]))

if __name__ == '__main__':
    main()