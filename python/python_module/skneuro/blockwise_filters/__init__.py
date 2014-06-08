from ..utilities import blockwiseCaller
from .. import denoising
from multiprocessing import cpu_count
import numpy
import vigra

def blockwiseGaussianSmoothing(image, sigma, out=None, blockShape=None, nThreads=None):
    shape = image.shape

    if blockShape is None:
        blockShape = [min(s, 100) for s in shape]

    if nThreads is None:
        nThreads = cpu_count()

    if out is None:
        out = numpy.empty(shape=shape,dtype=numpy.float32)

    margin = int(2.0*sigma +1.0+0.5)

    blockwiseCaller(f=denoising.gaussianSmoothing,margin=margin,shape=shape,
                    blockShape=blockShape, nThreads=nThreads,
                    inputKwargs=dict(image=image),
                    paramKwagrs=dict(sigma=sigma),
                    out=out)
    return out


def blockwiseGaussianGradientMagnitude(image, sigma, out=None, blockShape=None, nThreads=None):
    shape = image.shape
    image = vigra.taggedView(image, 'xyz')

    if blockShape is None:
        blockShape = [min(s, 100) for s in shape]

    if nThreads is None:
        nThreads = cpu_count()

    if out is None:
        out = numpy.empty(shape=shape,dtype=numpy.float32)

    margin = int(2.0*sigma +1.0+0.5)

    blockwiseCaller(f=vigra.filters.gaussianGradientMagnitude,margin=margin,shape=shape,
                    blockShape=blockShape, nThreads=nThreads,
                    inputKwargs=dict(volume=image),
                    paramKwagrs=dict(sigma=sigma),
                    out=out)
    return out