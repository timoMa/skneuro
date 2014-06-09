from skneuro import utilities
from skneuro import denoising
from skneuro import blockwise_filters
import numpy
import sys



def test_blockwise_gauss():

    shape = [100, 100, 100]
    blockShape = [32, 32, 32]

    totalData = numpy.random.rand(*shape).astype(numpy.float32)

    print "blockwiseGaussianSmoothing"
    result = blockwise_filters.blockwiseGaussianSmoothing(totalData,1.0,nThreads=3,blockShape=blockShape)
    print "blockwiseGaussianGradientMagnitude"
    result = blockwise_filters.blockwiseGaussianGradientMagnitude(totalData,1.0,nThreads=3,blockShape=blockShape)
    print "done"
    