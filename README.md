# pycuda-skeletonization

This repo is an implementation of skeletonization algorithm using pycuda; using the computation power of GPU
in python.

Skeletonization is a computationally expensive operation only based on iterative thinning algorithm which erodes to a single pixel thickness based on the distance transform

I have my pipeline running on python, but have an algorithm like skeletonization which is very complex propably requires some GPU power for faster computation. Pycuda gives an excellent interface to implement CUDA algorithms using python.


# Limitations

The current implementation is not useful for smaller objects as there is memory copy latency with the CPU and GPU. For much complicated images with lots of noise, this algorithm will be useful.


# Acknowledgements

The first reference started with skeleton implementation by scikit-learn and used most of the pycuda tutorials to actually have the code ready to work.
