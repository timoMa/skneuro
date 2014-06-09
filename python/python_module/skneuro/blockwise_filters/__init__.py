from ..utilities import blockwiseCaller as _bc
from .. import denoising
from multiprocessing import cpu_count
import numpy
import vigra



def _prepare(shape, blockShape=None, out=None, dtype=numpy.float32):
    if blockShape is None:
        blockShape = [min(s, 100) for s in shape]
    if out is None:
        out = numpy.empty(shape=shape, dtype=dtype)

    return blockShape, out


def blockwiseGaussianSmoothing(image, sigma, out=None, blockShape=None, nThreads=cpu_count()):
    blockShape, out = _prepare(image.shape, blockShape)
    margin = int(2.0*sigma + 1.0+0.5)

    func = denoising.gaussianSmoothing
    _bc(f=func, margin=margin, blockShape=blockShape, nThreads=nThreads, inputKwargs=dict(image=image),
        paramKwagrs=dict(sigma=sigma), out=out)
    return out


def blockwiseGaussianGradientMagnitude(image, sigma, out=None, blockShape=None, nThreads=None):
    blockShape, out = _prepare(image.shape, blockShape)
    margin = int(2.0*sigma + 1.0+0.5)
    func = vigra.filters.gaussianGradientMagnitude
    _bc(f=func, margin=margin, blockShape=blockShape, nThreads=nThreads, inputKwargs=dict(volume=image),
        paramKwagrs=dict(sigma=sigma), out=out)
    return out


def blockwiseLaplacianOfGaussian(image, scale, out=None, blockShape=None, nThreads=cpu_count()):
    blockShape, out = _prepare(image.shape, blockShape)
    margin = int(2.0*scale + 1.0+0.5)

    def func(image, scale, out=None):
        img = vigra.taggedView(image, 'xyz')
        out = vigra.filters.laplacianOfGaussian(img, scale=scale, out=out)
        return out

    _bc(f=func, margin=margin, blockShape=blockShape, nThreads=nThreads, inputKwargs=dict(image=image),
        paramKwagrs=dict(scale=scale), out=out)
    return out


def blockwiseStructureTensorSortedEigenvalues(image,  innerScale, outerScale, out=None, blockShape=None, nThreads=None):

    def func(image,  innerScale, outerScale, out=None):
        img = vigra.taggedView(image, 'xyz')
        out = vigra.filters.structreTensorEigenvalues(img, innerScale=innerScale, outerScale=outerScale, out=out)
        out = numpy.sort(out, axis=3)
        return out
    blockShape, out = _prepare(image.shape, blockShape)
    margin = int(2.0*max(innerScale, outerScale) + 1.0+0.5)

    _bc(f=func, margin=margin, blockShape=blockShape, nThreads=nThreads, inputKwargs=dict(volume=image),
        paramKwagrs=dict(innerScale=innerScale, outerScale=outerScale), out=out)
    return out


def blockwiseHessianOfGaussianSortedEigenvalues(image,  scale, out=None, blockShape=None, nThreads=None):

    def func(image,  scale, out=None):
        img = vigra.taggedView(image, 'xyz')
        out = vigra.filters.hessianOfGaussianEigenvalues(img, scale=scale, out=out)
        out = numpy.sort(out, axis=3)
        return out

    blockShape, out = _prepare(image.shape, blockShape)
    margin = int(2.0*scale + 1.0+0.5)

    _bc(f=func, margin=margin, blockShape=blockShape, nThreads=nThreads, inputKwargs=dict(volume=image),
        paramKwagrs=dict(scale=scale), out=out)
    return out


def binwiseDistaneTransform(image, minmax, bins):
    pass


def hist_plus_non_linear_smooth_stuff():
    pass
