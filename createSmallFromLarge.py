import sys
from subprocess import call
import glob

bigImages = glob.glob(sys.argv[1])

for bigImage in bigImages:
    parts = bigImage.split('/')
    directory = parts[0]
    imageNumber = parts[1].split('-')[-1]
    cmd = 'convert {0} -resize 2X2 {1}/small-{2}'.format(bigImage, directory, imageNumber)
    call(cmd, shell=True)
