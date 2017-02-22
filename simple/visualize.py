import os
from enhancer import *

VISUALIZE_FILE = os.getcwd()+'/visual.html'
DISPLAY_SINGLE_IMAGE = '<center><h1>{1}</h1><img src={0} width="500px"></center>'

def updateVisual(html, message='Please refresh the visualization page. Hit enter when ready to continue'):
    with open(VISUALIZE_FILE, 'w+') as output:
        output.write(html)

    if message != None:
        input(message)

def visualizeLayer(inputArray, outputArray, layerName):
    htmlOutput = '<center><h1>Visualizing Layer: {0}</h1>'.format(layerName)
    htmlOutput += fourDBlocktoList(inputArray, str(inputArray.shape), layerName + 'in')
    htmlOutput += '<h2 style=display:inline-block;margin:20px;">----></h2>'
    htmlOutput += fourDBlocktoList(outputArray, str(outputArray.shape), layerName + 'out')
    htmlOutput += '</center>'
    updateVisual(htmlOutput)

def fourDBlocktoList(block, dim, name):
    htmlOutput = '<ul style="list-style-type:none; padding: 5px;background-color: #eee;display: inline-block;">'
    htmlOutput += '<li>{0}</li>'.format(dim)
    threeDBlock = block[0] # This assumes the 'batch_size' dimension is 1
    ndx = 1
    for layer in threeDBlock:
        img = reversePreProcessImage(layer).convert('RGB')
        imageName = 'visualizeImages/'+name + (str(ndx)) + '.png'
        img.save(imageName)
        htmlOutput += '<li><h4 style="display: inline-block;margin:10px;">{1}</h4><img src={0} width="100px"></li>'.format(imageName,ndx)
        ndx += 1
    htmlOutput += '</ul>'
    return htmlOutput