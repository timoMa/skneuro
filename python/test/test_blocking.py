from skneuro import utilities
from skneuro import denoising
from skneuro import blockwise_filters
import numpy
import sys



shape = [400, 200, 200]
blockShape = [64, 64, 64]

totalData = numpy.random.rand(*shape).astype(numpy.float32)

print "blockwiseGaussianSmoothing"
result = blockwise_filters.blockwiseGaussianSmoothing(totalData,1.0,nThreads=3,blockShape=blockShape)
print "blockwiseGaussianGradientMagnitude"
result = blockwise_filters.blockwiseGaussianGradientMagnitude(totalData,1.0,nThreads=3,blockShape=blockShape)
print "done"
