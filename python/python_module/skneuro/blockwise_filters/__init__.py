from ..utilities import blockwiseCaller as _bc
from .. import denoising
from .. import parallel
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

gaussianSmoothing = blockwiseGaussianSmoothing


def truncatedDistanceTransform(image, truncateAt, background = True, out=None, blockShape=None, nThreads=cpu_count()):
    blockShape, out = _prepare(image.shape, blockShape)
    margin = int(truncateAt + 0.5)

    def func(image, truncateAt, background, out=None):
        imgv = vigra.taggedView(image, 'xyz')
        img = numpy.require(imgv, dtype='float32')
        out = vigra.filters.distanceTransform3D(img, background=background, out=out)
        out[numpy.where(out>truncateAt)] = truncateAt
        return out
    _bc(f=func, margin=margin, blockShape=blockShape, nThreads=nThreads, inputKwargs=dict(image=image),
        paramKwagrs=dict(truncateAt=truncateAt, background=background), out=out)
    return out



def gaussianRankOrder(image, ranks, sigmaS, sigmaB=1.0, bins=100,
                             blockShape=None, nThreads=cpu_count()):
    blockShape, out = _prepare(image.shape, blockShape, channels=len(ranks))
    margin = int(3.0*sigmaS + 0.5)

    sigmas = (sigmaS, sigmaS, sigmaS, sigmaB)

    minVal,maxVal = parallel.arrayMinMax(image)

    func = vigra.histogram.gaussianRankOrder
    _bc(f=func, margin=margin, blockShape=blockShape, nThreads=nThreads, inputKwargs=dict(image=image),
        paramKwagrs=dict(sigmas=sigmas,ranks=ranks,bins=bins,minVal=minVal,maxVal=maxVal), out=out)
    return out


def grayscaleDilation(image,sigma,blockShape=None, nThreads=cpu_count()):
    blockShape, out = _prepare(image.shape, blockShape)
    margin = 15

    func = vigra.filters.multiGrayscaleDilation
    _bc(f=func, margin=margin, blockShape=blockShape, nThreads=nThreads, inputKwargs=dict(image=image),
        paramKwagrs=dict(sigma=sigma), out=out)
    return out


def smoothedGrayscaleDilation(image,sigma,sigmaS,pre=True,post=False, blockShape=None, nThreads=cpu_count()):
    blockShape, out = _prepare(image.shape, blockShape)
    margin = 30

    def func(image, sigma,sigmaS,pre,post, out=None):
        img = vigra.taggedView(image, 'xyz')
        if pre:
            imgS = vigra.filters.gaussianSmoothing(img, sigma=sigmaS)
        else :
            imgS = img

        
        if post:
            imgD = vigra.filters.multiGrayscaleDilation(imgS, sigma=sigma)
            out = vigra.filters.gaussianSmoothing(imgD, sigma=sigmaS, out=out)
        else :
            out = vigra.filters.multiGrayscaleDilation(imgS, sigma=sigma, out=out)
        return out

    _bc(f=func, margin=margin, blockShape=blockShape, nThreads=nThreads, inputKwargs=dict(image=image),
        paramKwagrs=dict(sigma=sigma, sigmaS=sigmaS,pre=pre,post=post), out=out)
    return out


def grayscaleErosion(image,sigma,blockShape=None, nThreads=cpu_count()):
    blockShape, out = _prepare(image.shape, blockShape)
    margin = 15

    func = vigra.filters.multiGrayscaleErosion
    _bc(f=func, margin=margin, blockShape=blockShape, nThreads=nThreads, inputKwargs=dict(image=image),
        paramKwagrs=dict(sigma=sigma), out=out)
    return out

def grayscaleOpening(image,sigma,blockShape=None, nThreads=cpu_count()):
    blockShape, out = _prepare(image.shape, blockShape)
    margin = 15

    func = vigra.filters.multiGrayscaleOpening
    _bc(f=func, margin=margin, blockShape=blockShape, nThreads=nThreads, inputKwargs=dict(image=image),
        paramKwagrs=dict(sigma=sigma), out=out)
    return out

def grayscaleClosing(image,sigma,blockShape=None, nThreads=cpu_count()):
    blockShape, out = _prepare(image.shape, blockShape)
    margin = 15

    func = vigra.filters.multiGrayscaleClosing
    _bc(f=func, margin=margin, blockShape=blockShape, nThreads=nThreads, inputKwargs=dict(image=image),
        paramKwagrs=dict(sigma=sigma), out=out)
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






hessianOfGaussianLargestEigenvalues = blockwiseHessianOfGaussianLargestEigenvalues

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


