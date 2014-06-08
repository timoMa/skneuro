from skneuro import utilities
from skneuro import denoising
import numpy
import sys



shape = [1000, 200, 200]
blockShape = [100, 100, 100]

totalData = numpy.random.rand(*shape).astype(numpy.float32)
resultData = numpy.random.rand(*shape).astype(numpy.float32)


blockwiseCaller  = utilities.blockwiseCaller


blockwiseCaller( 
    f=denoising.gaussianSmoothing,
    margin=10,
    shape=shape,
    blockShape=blockShape,
    nThreads=4,
    inputKwargs=dict(image=totalData),
    paramKwagrs=dict(sigma=2.0),
    output=resultData
)
