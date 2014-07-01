from skneuro import utilities
from skneuro import denoising
from skneuro import blockwise_filters
import numpy
import sys
import vigra
from time import time






path = "/home/tbeier/Desktop/data.h5"

shape = [1000, 100, 1000]

data=numpy.random.rand(*shape).astype(numpy.float32)
shape = data.shape
print shape
blockShape = [100, 100, 100]





print "blockwiseGaussianGradientMagnitude"
t0=time()
result = blockwise_filters.blockwiseGaussianGradientMagnitude(data, 5.0, nThreads=22, blockShape=blockShape)
t1=time()
print "done in",t1-t0

tp = t0-t1

print "blockwiseGaussianGradientMagnitude"
t0=time()
result = vigra.filters.gaussianGradientMagnitude(data, 5.0)
t1=time()
print "done in",t1-t0

ts = t0-t1


print ts/tp
