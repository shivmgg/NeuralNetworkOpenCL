# Neural Network Implementation in OpenCL
This is an OpenCL implementation of VGG-13 architecture for CIFAR-10 dataset. 

Modules implemented:
- Linear: Implements matrix multiplication operation between weights and input layers. Gives
output after applying HardTanh operation.
- BatchNorm1D: Standard 1d batchnorm function
- Conv2D: Using output data decomposition to parallelize the convolution operation between
input and weight kernels.
- MaxPool2D: Similar to Conv2D
- LogSoftMax: Each work-item calculates the total sum of exponentials followed by its respective
output

I tested my results on NVIDIA GeForce 1050M. It is a 4 GB mobile desktop GPU. Since my
architecture has around 10 Conv2D layers and 3 FC layers, it took around 0.47220 seconds to
run inference on a single image.

Implementation details:

- I used 3D images to store my inputs and weights. Utilizing output data decomposition to calculate the output products in parallel.
- By looking at the specification of my device, I decided to store my weights in (KxK)xCinxCout format while input is stored in MxNxCin format.
- Moreover, to avoid external padding in convolution operation, indices are carefully defined carefully.
