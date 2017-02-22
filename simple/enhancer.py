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

from visualize import *

import numpy as np
import theano, theano.tensor as T
import lasagne
from lasagne.layers import Conv2DLayer as ConvLayer, Deconv2DLayer as DeconvLayer, Pool2DLayer as PoolLayer
from lasagne.layers import InputLayer, ConcatLayer, ElemwiseSumLayer, batch_norm, ParametricRectifierLayer

DEBUG = False

os.environ.setdefault('THEANO_FLAGS', 'exception_verbosity=high')

def error(message, errorCode=-1):
    print('ERROR: ' + message + ' EXITING.')
    sys.exit(errorCode)

def debug(message):
    if (DEBUG):
        print('DEBUG: ' + message)

def parseArgs():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--train', default=False, action='store_true')
    parser.add_argument('--target-images', default='./dataset/big*.png')
    parser.add_argument('--source-images', default='./dataset/small*.png')
    parser.add_argument('--target-image-size', default=16)
    parser.add_argument('--source-image-size', default=8)
    
    parser.add_argument('--image', '-i', default=None)
    parser.add_argument('--channels', default=3)
    
    parser.add_argument('--visualize', '-v', default=False, action='store_true')
    parser.add_argument('--debug', '-d', default=False, action='store_true')

    args = parser.parse_args()

    if args.visualize:
        print('-- Visual mode enabled. --')
        print('Please open the following link with a browser.')
        updateVisual('<h1>Visual Mode Enabled</h1>', message=None)
        print(VISUALIZE_FILE)

    if args.debug:
        global DEBUG
        DEBUG = True

    return args

# The network object is made up of three main components.
class Network(object):

    def __init__(self, args): 
            self.generatorNetwork = self.initializeGeneratorNetwork(args)

    def initializeGeneratorNetwork(self, args):
        network = collections.OrderedDict()
        input_var = T.tensor4('input')
        network['input'] = InputLayer((1, args.channels, args.source_image_size, args.source_image_size), input_var=input_var)
        network['conv_1'] = ConvLayer(network['input'], 16, (3,3), stride=1, pad=1, nonlinearity=None)
        network['prelu_1'] = ParametricRectifierLayer(network['conv_1'])
        return network

    # Given an image, generate an image of double the size
    def generate(self, image, args):
        network = self.generatorNetwork

        layerInput = image
        for layer in network:
            debug('Processing layer with name "{0}" and input with dimensions {1}'.format(layer, layerInput.shape))
            outputFunction = lasagne.layers.get_output(network[layer], layerInput)
            output = outputFunction.eval()
            visualizeLayer(layerInput, output, layer)
            layerInput = output

# Takes an image, loaded with PIL (Image.open) and processes it
# so it is ready to be inputed to lasagne and theano
def preProcessImage(image, args):
    # Convert the image to an array of floats, squash colors values to be in the range 0 to 1.
    img = np.asarray(image, dtype=np.float64) / 255.

    # Flip the image, so it's BGR, not RGB, and make sure it's the correct size.
    img = img.transpose(2,0,1).reshape(3, args.source_image_size, args.source_image_size)

    # Wrap as a 4d array and return.
    return np.asarray([img])

# Reverse of the preProcessing doen above. This is so we can convert intermediate data frm
# the middle of a network to an image for visualization purposes.
def reversePreProcessImage(array):
    array = array * 255.

    return Image.fromarray(array)

def main():
    args = parseArgs()

    if args.train:
        error('not implemented yet')
    else:
        network = Network(args)
        img = Image.open(args.image)
        if args.visualize:
            updateVisual(DISPLAY_SINGLE_IMAGE.format(args.image, 'Input {0}'.format(img.size)))

        img = preProcessImage(img, args)
        network.generate(img, args)
     
if __name__ == '__main__':
    main()


    # compare image1 image2 -compose src diff.png
    