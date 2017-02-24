import numpy as np
from PIL import Image
import sys

def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

def main():
    weights = np.load(sys.argv[2])
    src = sys.argv[1]
    img = np.asarray(Image.open(src).convert('RGB')).transpose(2,0,1).flatten()/255.
    predictedOut = (nonlin(np.dot(img,weights))*255).reshape(3, 4, 4).astype('uint8').transpose(1,2,0)
    Image.fromarray(predictedOut).save("OUT.png")

if __name__ == '__main__':
    main()