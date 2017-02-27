import numpy as np
from PIL import Image
import sys

def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

def main():
    src = sys.argv[1]
    img = np.asarray(Image.open(src).convert('RGB')).transpose(2,0,1)

    topLeft = []
    for x in img:
        imgParts = x[np.ix_([0,1],[0,1])]
        topLeft.append(imgParts)

    topRight = []
    for x in img:
        imgParts = x[np.ix_([0,1],[2,3])]
        topRight.append(imgParts)

    bottomLeft = []
    for x in img:
        imgParts = x[np.ix_([2,3],[0,1])]
        bottomLeft.append(imgParts)

    bottomRight = []
    for x in img:
        imgParts = x[np.ix_([2,3],[2,3])]
        bottomRight.append(imgParts)

    topLeft = np.asarray(topLeft)
    topRight = np.asarray(topRight)
    bottomLeft = np.asarray(bottomLeft)
    bottomRight = np.asarray(bottomRight)
    topLeft = topLeft.astype('uint8').transpose(1,2,0)
    topRight = topRight.astype('uint8').transpose(1,2,0)
    bottomLeft = bottomLeft.astype('uint8').transpose(1,2,0)
    bottomRight = bottomRight.astype('uint8').transpose(1,2,0)

    Image.fromarray(topLeft).save("topleft.png")
    Image.fromarray(topRight).save("topright.png")
    Image.fromarray(bottomLeft).save("bl.png")
    Image.fromarray(bottomRight).save("br.png")

    upperHalf = np.hstack((topLeft, topRight))
    lowerHalf = np.hstack((bottomLeft, bottomRight))

    original = np.vstack((upperHalf, lowerHalf))

    Image.fromarray(original).save("re.png")


    

if __name__ == '__main__':
    main()