# Enhancer
Enhancer is a simple two layer neural network used to 'enhance' an image. This is done by sliding a 2x2 window over an image, expanding each tile into a 4x4. 

The image below, shows how a 2x2 rgb image is converted into it's input format.

![ScreenShot](/examples/Input.png) 

The input passes through 2 fully connected layers, pictured below.

![ScreenShot](/examples/Network.png) 

## Examples

An enhancment of a picture of my dog, Bodhi.

![ScreenShot](/examples/Bodhi.png)
![ScreenShot](/examples/BodhiEye.png) 

An enhancment of a human face.

![ScreenShot](/examples/Face.png) 
