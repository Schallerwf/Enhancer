import numpy as np
from PIL import Image
from PIL import ImageOps
import sys
from math import sqrt
from subprocess import call

def error(message):
    print('ERROR: {0}'.format(message))
    sys.exit(-1)

def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

def rolling_window(arr, window):
    """Very basic multi dimensional rolling window. window should be the shape of
    of the desired subarrays. Window is either a scalar or a tuple of same size
    as `arr.shape`.
    """
    shape = np.array(arr.shape*2)
    strides = np.array(arr.strides*2)
    window = np.asarray(window)
    shape[arr.ndim:] = window # new dimensions size
    shape[:arr.ndim] -= window - 1
    if np.any(shape < 1):
        raise ValueError('window size is too large')
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

def main():
    src = sys.argv[1]
    weights = np.load(sys.argv[2])
    img = np.asarray(Image.open(src).convert('RGB')).transpose(2,0,1)
    imgWidth = img[0].shape[0]
    imgHeight = img[0].shape[1]

    cropSize = sqrt(weights.shape[0] / 3)
    print("Weight dimensions: " + str(weights.shape[0]) + ", " + str(weights.shape[1]))
    print("Crop size: " + str(cropSize))
    print("Image size: " + str(imgWidth) + ", " + str(imgHeight))
    print("Tile size: " + str(cropSize) + ", " + str(cropSize))
    print("Num of tiles: " + str((imgWidth * imgHeight) // (cropSize * cropSize)))
    resultSize = cropSize * 2

    # Testing if proper image size
    if imgHeight != imgWidth:
        error('Input image must be square.')

    if imgWidth % cropSize != 0:
        error('Input image must be divisible by multiplication factor.')

    # Crop the image into various tiles
    # num_tiles = (imgWidth * imgHeight) // (cropSize * cropSize)
    # image_slicer.slice(sys.argv[1], num_tiles)

    print("Image array: " + str(img.shape))

    # Create the array to hold the properly sized result
    result = np.zeros(shape=[imgWidth * 2, imgHeight * 2, 3])

    # Get all possible tiles
    slices = rolling_window(img, (3, 2, 2))

    # Apply weights, reshape into resultSize x resultSize
    d = 0
    for y in range(0, slices.shape[2]):
        c = 0
        for x in range(0, slices.shape[1]):
            tile = np.squeeze(slices[:, x, y]).flatten()/255.    # convert 4D to 3D
            expandedTile = (nonlin(np.dot(tile,weights))*255).reshape(3, resultSize, resultSize).astype('uint8').transpose(1,2,0)
            
            # Add overlapping values into the result array
            result[c:c+resultSize, d:d+resultSize, :] += expandedTile
            c = c + 2
        d = d + 2
            
    # Average overlapped values and save the new image
    result[1:(imgWidth*2 - 2), 1:(imgHeight*2 - 2), :] //= 3
    Image.fromarray(np.uint8(result), 'RGB').save("double.png")

if __name__ == '__main__':
    main()