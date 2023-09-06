This directory contains some data extracted from the MNIST image
dataset but is presented in textural format.  Each image appears on a
singe line as.

<lab> : <pixel1> <pixel2> <pixel3> ... <pixel784>

The images are 28x28 for 784 pixels and the intensities are encoded as
integers from 0 and 255 (0 for black, 255 for white).

For example, the first few lines of 10.txt show

7 :   0   0   0   0   0 ... 84 185 159 151  60  36   0   0 ...
2 :   0   0   0   0   0 ...  0  77 251 210  25   0   0   0 ...
1 :   0   0   0   0   0 ...  0   0   0   0   0   0   0   0 ...
0 :   0   0   0   0   0 ...  0   0   0   0 110 190 251 251 ...

which are images of a hand written 7, 2, 1, and 0 respectively.

The format is chosen to make it reasonably easy to parse into data
structures and arrays for Machine learning applications and to slice
files using text tools such as head / tail / awk.

