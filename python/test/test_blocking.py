from skneuro import utilities
from skneuro import denoising
from skneuro import blockwise_filters
import numpy
import sys
import psutil


def test_blockwise():

    shape = [1000, 1000, 400]
    blockShape = [100, 100, 100]

    totalData = numpy.random.rand(*shape).astype(numpy.float32)

    print "blockwiseGaussianSmoothing"
    result = blockwise_filters.blockwiseGaussianSmoothing(totalData,1.0,nThreads=12,blockShape=blockShape)
    for x in range(20):
        blockwise_filters.blockwiseGaussianGradientMagnitude(totalData,1.0,nThreads=12,blockShape=blockShape)

test_blockwise()
