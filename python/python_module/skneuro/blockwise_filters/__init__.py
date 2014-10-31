from ..utilities import blockwiseCaller as _bc
from .. import denoising
from multiprocessing import cpu_count
import numpy
import vigra



def _prepare(shape, blockShape=None, out=None, dtype=numpy.float32, 
             channels=1):
    if blockShape is None:
        blockShape = [min(s, 100) for s in shape]
    if out is None:
        if channels>1:
            s = tuple(shape)+(channels,)
        else :
            s = shape
        out = numpy.empty(shape=s, dtype=dtype)

    return blockShape, out


def blockwiseGaussianSmoothing(image, sigma, out=None, blockShape=None, nThreads=cpu_count()):
    blockShape, out = _prepare(image.shape, blockShape)
    margin = int(4.0*sigma + 1.0+0.5)

    func = denoising.gaussianSmoothing
    _bc(f=func, margin=margin, blockShape=blockShape, nThreads=nThreads, inputKwargs=dict(image=image),
        paramKwagrs=dict(sigma=sigma), out=out)
    return out



def blockwiseMedianSmoothing(image, radius, mode='reflect', cval=0.0, origin=0,
                             blockShape=None, nThreads=cpu_count()):
    blockShape, out = _prepare(image.shape, blockShape)
    margin = int(radius + 1)

    func = denoising.medianSmoothing
    _bc(f=func, margin=margin, blockShape=blockShape, nThreads=nThreads, inputKwargs=dict(image=image),
        paramKwagrs=dict(radius=radius, mode=mode, cval=cval, origin=origin), out=out)
    return out



def blockwiseMultiGrayscaleDilation(image, sigma, out=None, blockShape=None, nThreads=None):
    blockShape, out = _prepare(image.shape, blockShape)
    margin = int(3.0*sigma + 1.0+0.5)
    func = vigra.filters.multiGrayscaleDilation
    _bc(f=func, margin=margin, blockShape=blockShape, nThreads=nThreads, inputKwargs=dict(volume=image),
        paramKwagrs=dict(sigma=sigma), out=out)
    return out


def blockwiseGaussianGradient(image, sigma, out=None, blockShape=None, nThreads=None):
    blockShape, out = _prepare(image.shape, blockShape, channels=len(image.shape))
    margin = int(3.0*sigma + 1.0+0.5)
    func = vigra.filters.gaussianGradient
    _bc(f=func, margin=margin, blockShape=blockShape, nThreads=nThreads, inputKwargs=dict(volume=image),
        paramKwagrs=dict(sigma=sigma), out=out)
    return out





def blockwiseGaussianGradientMagnitude(image, sigma, out=None, blockShape=None, nThreads=None):
    blockShape, out = _prepare(image.shape, blockShape)
    margin = int(3.0*sigma + 1.0+0.5)
    func = vigra.filters.gaussianGradientMagnitude
    _bc(f=func, margin=margin, blockShape=blockShape, nThreads=nThreads, inputKwargs=dict(volume=image),
        paramKwagrs=dict(sigma=sigma), out=out)
    return out


def blockwiseLaplacianOfGaussian(image, scale, out=None, blockShape=None, nThreads=cpu_count()):
    blockShape, out = _prepare(image.shape, blockShape)
    margin = int(3.0*scale + 1.0+0.5)

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
        out = vigra.filters.structureTensorEigenvalues(img, innerScale=innerScale, outerScale=outerScale, out=out).squeeze()
        out = numpy.sort(out, axis=3)
        return out
    blockShape, out = _prepare(image.shape, blockShape, channels=3)
    margin = int(3.0*max(innerScale, outerScale) + 1.0+0.5)

    _bc(f=func, margin=margin, blockShape=blockShape, nThreads=nThreads, inputKwargs=dict(image=image),
        paramKwagrs=dict(innerScale=innerScale, outerScale=outerScale), out=out)
    return out


def blockwiseHessianOfGaussianSortedEigenvalues(image,  scale, out=None, blockShape=None, nThreads=None):

    def func(image,  scale, out=None):
        fImg  = numpy.require(image, dtype=numpy.float32)
        fImg = vigra.taggedView(fImg, 'xyz')
        out = vigra.filters.hessianOfGaussianEigenvalues(fImg, scale=scale, out=out)
        out = numpy.sort(out, axis=3)
        return out

    blockShape, out = _prepare(image.shape, blockShape, channels=3)
    margin = int(3.0*scale + 1.0+0.5)

    _bc(f=func, margin=margin, blockShape=blockShape, nThreads=nThreads, inputKwargs=dict(image=image),
        paramKwagrs=dict(scale=scale), out=out)
    return out


def blockwiseHessianOfGaussianLargestEigenvalues(image,  scale, out=None, blockShape=None, nThreads=None):
    #print "total shape", image.shape
    def func(image,  scale, out=None):
        #print "imageShape", image.shape
        fImg  = numpy.require(image, dtype=numpy.float32)
        fImg = vigra.taggedView(fImg, 'xyz')
        out = vigra.filters.hessianOfGaussianEigenvalues(fImg, scale=scale, out=out)
        out = numpy.max(out, axis=3)
        return out

    blockShape, out = _prepare(image.shape, blockShape)
    margin = int(3.0*scale + 1.0+0.5)
    _bc(f=func, margin=margin, blockShape=blockShape, nThreads=nThreads, inputKwargs=dict(image=image),
        paramKwagrs=dict(scale=scale), out=out)
    return out


def binwiseDistaneTransform(image, minmax, bins):
    pass


def hist_plus_non_linear_smooth_stuff():
    pass



def smooth_neuron_prob_map():
    pass

def smooth_mitochondria_prob_map():
    pass



def bin_wise_non_local_mean():
    pass
