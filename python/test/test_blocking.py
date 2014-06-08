from skneuro import utilities
from skneuro import denoising
from skneuro import blockwise_filters
import numpy
import sys



shape = [400, 200, 200]
blockShape = [100, 100, 100]

totalData = numpy.random.rand(*shape).astype(numpy.float32)
resultData = numpy.random.rand(*shape).astype(numpy.float32)


result = blockwise_filters.blockwiseGaussianSmoothing(totalData,1.0,nThreads=3)
result = blockwise_filters.blockwiseGaussianGradientMagnitude(totalData,1.0,nThreads=3)

