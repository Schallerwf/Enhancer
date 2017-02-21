import io
import os
import sys
import argparse
import glob
from subprocess import call
from PIL import Image
import random
import pickle
import collections
import bz2
import itertools
import scipy

import numpy as np
import theano, theano.tensor as Theano
import lasagne
from lasagne.layers import Conv2DLayer as ConvLayer, Deconv2DLayer as DeconvLayer, Pool2DLayer as PoolLayer
from lasagne.layers import InputLayer, ConcatLayer, ElemwiseSumLayer, batch_norm

DEBUG = True

#DEBUG 
os.environ.setdefault('THEANO_FLAGS', 'exception_verbosity=high')

def error(message, errorCode=-1):
    print('ERROR: ' + message)
    sys.exit(errorCode)

def debug(message):
    if (DEBUG):
        print('DEBUG: ' + message)

def parseArgs():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--target-images', default='./dataset/big*.png')
    parser.add_argument('--source-images', default='./dataset/small*.png')
    parser.add_argument('--pretrained-model', default=None)
    parser.add_argument('--source-image-size', default=8)
    parser.add_argument('--target-image-size', default=16)
    return parser.parse_args()

def loadDataset(target_images, source_images):
    # Get all matching files
    targets = glob.glob(target_images)
    sources = glob.glob(source_images)

    if len(targets) != len(sources):
        error('Must have same number of target and source images')
    else:
        debug('Loading {0} images'.format(len(targets)*2))

    # For each image in the list of images, load it into an array, and
    # add it to the total array.
    targetsArray = np.array([np.array(Image.open(image)) for image in targets])
    sourcesArray = np.array([np.array(Image.open(image)) for image in sources])
    return (targetsArray, sourcesArray)

def extend(lst): 
    return itertools.chain(lst, itertools.repeat(lst[-1]))

def main():
    args = parseArgs()
    targets, source = loadDataset(args.target_images, args.source_images)
    network = Network(model=args.pretrained_model)

    img = scipy.ndimage.imread('./dataset/small-1.png', mode='RGB')
    out = network.processImage(img)
    out.save('OUTPUT.png')
    print(flush=True)

# Taken from https://github.com/alexjc/neural-enhance/blob/master/enhance.py#L228
class SubpixelReshuffleLayer(lasagne.layers.Layer):
    """Based on the code by ajbrock: https://github.com/ajbrock/Neural-Photo-Editor/"""

    def __init__(self, incoming, channels, upscale, **kwargs):
        super(SubpixelReshuffleLayer, self).__init__(incoming, **kwargs)
        self.upscale = upscale
        self.channels = channels

    def get_output_shape_for(self, input_shape):
        def up(d): return self.upscale * d if d else d
        return (input_shape[0], self.channels, up(input_shape[2]), up(input_shape[3]))

    def get_output_for(self, input, deterministic=False, **kwargs):
        out, r = Theano.zeros(self.get_output_shape_for(input.shape)), self.upscale
        for y, x in itertools.product(range(r), repeat=2):
            out=Theano.inc_subtensor(out[:,:,y::r,x::r], input[:,r*y+x::r*r,:,:])
        return out

class Network(object):

    def __init__(self, model=None): 
            self.network = collections.OrderedDict()
            self.network['target'] = InputLayer((None, 3, None, None))
            self.network['source'] = InputLayer((None, 3, None, None))

            # If a model is provided, use its pretrained weights and params, otherwise, start from scratch.
            config, params = {}, {} if model == None else self.loadPretrainedModel(model)

            self.setup_generator(self.getLastLayer())
            #self.printNetwork()

            #CorrMM images and kernel must have the same stack size

            input_tensor, seed_tensor = Theano.tensor4(), Theano.tensor4()
            input_layers = {self.network['target']: input_tensor, self.network['source']: seed_tensor}
            for layer in ['source', 'out']:
                debug('Processing layer ' + layer)
                output = lasagne.layers.get_output(self.network[layer], input_layers, deterministic=True)

            self.predict = theano.function([seed_tensor], output)

    def printNetwork(self):
        for layer in self.network:
            print(layer)

    def getLastLayer(self):
        return list(self.network.values())[-1]

    def loadPretrainedModel(self, model):
        return pickle.load(bz2.open(model, 'rb'))

    def make_layer(self, name, previousLayer, num_filters, filter_size=(3,3), stride=(1,1), pad=(2,2), alpha=0.25):
        conv = ConvLayer(previousLayer, num_filters, filter_size, stride=stride, pad=pad, nonlinearity=None)
        prelu = lasagne.layers.ParametricRectifierLayer(conv, alpha=lasagne.init.Constant(alpha))
        self.network[name+'x'] = conv # A convolution is the result of passing a kernel through the filters
        self.network[name+'>'] = prelu # A prelu is a simplification of the current results in order to prevent
                                       # anything from getting too extreme (infinity values)
        return prelu

    def make_block(self, name, input, units):
        self.make_layer(name+'-A', input, units, alpha=0.1)
        return ElemwiseSumLayer([input, self.getLastLayer()])

    def processImage(self, originalImage):

        s = 2
        p = 0
        z = 2

        output = np.zeros((originalImage.shape[0] * z, originalImage.shape[1] * z, 3), dtype=np.float32)
        debug('Converting an image of dimensions {0}, to {1}'.format(originalImage.shape, output.shape))

        image = [np.transpose(originalImage)] # make the input 4d, where the 4th dimnsion is a batch size of 1
        
        # TODO - figure out return format here
        b = self.predict(image)
        b = np.transpose(b[0])
        print('.', end='', flush=True)
        output = b.clip(0.0, 1.0) * 255.0
        return scipy.misc.toimage(output, cmin=0, cmax=255)




    def setup_generator(self, input):
        #TODO change these values to be default aurguments
        num_filters = 64
        num_layers = 4
        upscale = 2

        # Creates convolutional layers, each one recieving input from the one previous
        self.make_layer('iter.0', input, num_filters, filter_size=(4,4), pad=(2,2))
        for i in range(0, num_layers):
            self.make_block('iter.%i'%(i+1), self.getLastLayer(), num_filters)

        # Creates upscaling layers, that increase the dimensions of the matrix.
        for i in range(0, upscale):
            self.make_layer('upscale%i.2'%i, self.getLastLayer(), num_filters*4)
            self.network['upscale%i.1'%i] = SubpixelReshuffleLayer(self.getLastLayer(), num_filters, 2)

        self.network['out'] = ConvLayer(self.getLastLayer(), 3, filter_size=(4,4), pad=(2,2), nonlinearity=None)

if __name__ == '__main__':
    main()