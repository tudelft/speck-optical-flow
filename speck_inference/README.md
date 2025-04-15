# Speck_Optical_Flow
This repository aim is to capture all the work done to get optical flow vectors using SPECK™ devkit. To run first installation of sinabs is required.

## Data Collection
The folder Data_Collection contains the code `data_collection_csv_new.py` required to collect data using speck2e devkit. This code is not available on sinabs(Deep Learning library for Spiking Neural Network for speck) site. 

## Optical Flow Requirements
The optical flow inference leverages the contrast maximization framework that uses Self-supervised learning with iterative warping [tincymax repo](https://github.com/Huizerd/tinycmax/tree/main). The framework contains encoder which downsamples the input data from (2, 128, 128) to (64, 16, 16), memory which Convolutional GRU and decoder which upsamples with bilinear interpolation. 

In speck, the complete framework cannot be put. However to make use of low latency, low power and low memory, simplified encoder is put on speck with ANN - SNN conversion. 

The spike outputs are then converted to continuous floating point values using gaussian smoothening and then send the remaining network.

The complete inference is done live with incoming inputs and output optical flows. For this run code, `speck_inference_64ch.py`. 

All the required files are in Optical_Flow_run folder.





