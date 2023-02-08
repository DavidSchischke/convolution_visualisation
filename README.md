# What does a Convolution see? 

This repo is contains code that visualizes the activation of different convolutions for a set of given inputs. It uses a 
**LeNet5** architecture with **MNIST** data in **PyTorch**. This is not a very sophisticated script, but something I was
interested in after watching a [3blue1brown video on Convolutions](https://youtu.be/KuXjwB4LzSA). I figured it might be 
interesting to some other people as well.

## Results

The LeNet-architecture ([LeCun et al., 1998](https://ieeexplore.ieee.org/abstract/document/726791)) utilizes two sets of 
small convolutions as the first two layers, making it ideal for this approach. The final model has an accuracy of 99.1%, 
so I assume the output of the Convolutions is fairly "helpful" for the classification task at hand. **All results can be 
found in the `media` folder** - in addition to the gifs in this readme, every slide of the gifs is also available as a 
.png file. All functions can be found in the (`visualization_functions.py`)[https://github.com/daaawit/convolution_visualisation/blob/main/visualization_functions.py] file

### `conv1` layer

This is the first layer, directly parsing the input. The resulting activations are still relatively similar to the 
original input: 

![](https://github.com/daaawit/convolution_visualisation/blob/main/media/conv1.gif)

If we blur the input a bit, we can see that the different convolutions "recognize" different parts of the shapes (light values
indicating high activation, dark values indicating low activation).

![](https://github.com/daaawit/convolution_visualisation/blob/main/media/conv1_blur.gif)

### `conv2` layer

The `conv2` layer comes after `conv1` -> `RelU`. It is still apparent that the different filters search for different shapes. However,
the shapes appear to be more general, with e.g. some convolutions showing high activation for diagonal inputs (especially apparent for
input "7". Again, a slight blur has been applied to help find the underlying shapes instead of focussing on individual pixels: 

![](https://github.com/daaawit/convolution_visualisation/blob/main/media/conv2_blur.gif)



### Training details

The model definition can be obtained from
the file [`data and models.py`](https://github.com/daaawit/convolution_visualisation/blob/main/data_and_model.py#L8). 

I trained the model for 30 epochs with SGD, an initial LR of 0.03, momentum of 0.9 and a Plateau Scheduler, resulting in 
a final test accuracy of 99.1%. The training script can be found at [`train_lenet.py`]. The trained model is available
as a `.pth` file: [`lenet5_trained.pth`](https://github.com/daaawit/convolution_visualisation/blob/main/lenet5_trained.pth).


## Requirements 

The code in this repository is fully executable. While I did not write a conda env file, the requirements are minimal: 

* `Matplotlib==3.6.3`
* `PyTorch==1.13.1`
* `Torchvision==0.14.1`

Additionally, if you want to render out gifs, you will need `imageio==2.25.0`.