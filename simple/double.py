import numpy as np
from PIL import Image
import sys
from math import sqrt
from subprocess import call

def error(message):
    print 'ERROR: {0}'.format(message)
    sys.exit(-1)

def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))


# Takes a 4x4 image, and generates an 8x8
def main():
    src = sys.argv[1]
    weights = np.load(sys.argv[2])
    img = np.asarray(Image.open(src).convert('RGB')).transpose(2,0,1)
    imgWidth = img[0].shape[0]
    imgHeight = img[0].shape[1]

    cropSize = sqrt(weights.shape[0] / 3)
    resultSize = cropSize * 2

    if imgHeight != imgWidth:
        error('Input image must be square.')

    if imgWidth % cropSize != 0:
        error('Input image must be divisible by multiplication factor.')

    tileCmd = 'convert {0}  +gravity -crop {1}x{1}  tiles/%d.png'.format(src, cropSize)
    call([tileCmd], shell=True)

    tilesPerRow = int(imgWidth // cropSize)

    rows = None
    for y in xrange(0, tilesPerRow):
        row = None
        for x in xrange(0, tilesPerRow):
            tile = np.asarray(Image.open('tiles/{0}.png'.format((y*tilesPerRow) + x)).convert('RGB')).transpose(2,0,1).flatten()/255.
            expandedTile = (nonlin(np.dot(tile,weights))*255).reshape(3, resultSize, resultSize).astype('uint8').transpose(1,2,0)

            if row == None:
                row = expandedTile.copy()
            else:
                row = np.hstack((row, expandedTile.copy()))
        if rows == None:
            rows = row.copy()
        else:
            rows = np.vstack((rows, row.copy()))

    Image.fromarray(rows).save("double.png")


if __name__ == '__main__':
    main()