from _denoising import *
from functools import partial
from scipy.ndimage.filters import median_filter

try:
    from skimage.filter import denoise_tv_bregman,denoise_tv_chambolle
except:
    from skimage.restoration import denoise_tv_bregman,denoise_tv_chambolle

import numpy
import vigra 

def nonLocalMean():
    pass


def gaussianSmoothing(image,sigma):
    inImg = numpy.require(image, dtype=numpy.float32)
    inImg = vigra.taggedView(inImg, 'xyz')
    #print inImg.shape, inImg.dtype
    return vigra.gaussianSmoothing(inImg, sigma=sigma)


def medianSmoothing(image,radius=None, footprint=None, mode='reflect', cval=0.0, origin=0, out=None):
    size = 2*radius+1
    size = [size]*3
    res = median_filter(input=image, size=size, footprint=footprint, output=out, mode=mode, cval=cval, origin=origin)
    if out is None:
        return res
    else:
        out[:] = res[:]
        return res

def tvBregman(image, weight, eps=0.001, maxIter=None, isotropic=True, out=None):

    image -= image.min()
    image /= image.max()

    kwargs = dict(image=image, weight=weight, eps=eps, isotropic=isotropic)
    if maxIter is not None :
        kwargs['maxIter']=maxIter

    res = denoise_tv_bregman(**kwargs)
    if out is None:
        return res
    else :
        out[:] = res[:]
        return out

def tvChambolle(image, weight, eps=0.001, maxIter=None, out=None):

    image -= image.min()
    image /= image.max()

    kwargs = dict(im=image, weight=weight, eps=eps, multichannel=False)
    if maxIter is not None :
        kwargs['n_iter_max']=maxIter

    res = denoise_tv_chambolle(**kwargs)
    if out is None:
        return res
    else :
        out[:] = res[:]
        return out

def anisotropicSmoothing():
    # impl "Edge Aware Anisotropic Diffusion for 3D Scalar Data"
    pass


def bilateralFiltering():
    pass


def smartSmoothing():
    # gauss with 0.75 first (no visible changes)
    # then nonLocalMean     (no assumption smoothing un)
    # then tv
    pass




def guidedFilter(image, guidanceImage, epsilon,
                 fMean=None):
    # 1) mean an corr
    print "mean 1"
    meanI = fMean(image)

    print "mean 2"
    meanG = fMean(guidanceImage)

    print "mean 3"
    corrG  = fMean(guidanceImage*guidanceImage)

    print "mean 4"
    corrGI = fMean(guidanceImage*image)

    # 2) var and cov    
    varG = corrG - meanG*meanG
    covGI = corrGI - meanI*meanG

    # 3) "a" and "b" image (as in paper)
    a = covGI / (varG + epsilon)
    b = meanI - a*meanG

    # 4) means on "a" and "b"
    print "mean 5"
    meanA = fMean(a)

    print "mean 6"
    meanB = fMean(b)

    # 5) make result q 
    q = meanA.view(numpy.ndarray)*guidanceImage.view(numpy.ndarray) + meanB.view(numpy.ndarray)

    return q


def gaussianGuidedFilter(image, guidanceImage, sigma, epsilon=0.4**2):
    meanFunction = partial(gaussianSmoothing,sigma=sigma)
    return guidedFilter(image, guidanceImage, epsilon, meanFunction)

def medianGuidedFilter(image, guidanceImage, radius, epsilon=0.4**2):
    meanFunction = partial(medianSmoothing,radius=radius)
    return guidedFilter(image, guidanceImage, epsilon, meanFunction)


if __name__ == "__main__":
    import numpy
    import vigra

    r=numpy.random.rand(30,30,30).astype(numpy.float32)
    r=vigra.taggedView(r,'xyz')
    res = medianSmoothing(r,radius=3)