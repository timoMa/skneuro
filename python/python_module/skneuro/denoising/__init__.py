import numpy
import vigra

from _denoising import *
from functools import partial
from scipy.ndimage.filters import median_filter

try:
    from skimage.filter import denoise_tv_bregman, denoise_tv_chambolle
except:
    from skimage.restoration import denoise_tv_bregman, denoise_tv_chambolle

def nonLocalMean(
    image,
    patchRadius=2,
    searchRadius=10,
    gamma=10,
):
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

    # to restore min max ?
    restore=False
    if image.min()<-1.0 or image.max()>1.0:
        restore=True
        gaussSmoothed = gaussianSmoothing(image,simga=1.0)
        oldMin = gaussSmoothed.min()
        oldMax = gaussSmoothed.max()
        image-=image.min()
        image/=image.max()


    kwargs = dict(image=image, weight=weight, eps=eps, isotropic=isotropic)
    if maxIter is not None :
        kwargs['maxIter']=maxIter

    res = denoise_tv_bregman(**kwargs)

    if restore:
        res -= res.min()
        res /= res.max()
        res *= (oldMax-oldMin)
        res += oldMin

    if out is None:
        return res
    else :
        out[:] = res[:]
        return out


def tvChambolle(image, weight, eps=0.001, maxIter=None, out=None):

    # to restore min max ?
    restore=False
    if image.min() < 1.0 or image.max() > 1.0:
        restore=True
        gaussSmoothed = gaussianSmoothing(image, simga=1.0)
        oldMin = gaussSmoothed.min()
        oldMax = gaussSmoothed.max()
        image -= image.min()
        image /= image.max()

    kwargs = dict(im=image, weight=weight, eps=eps, multichannel=False)

    if maxIter is not None:
        kwargs['n_iter_max'] = maxIter

    res = denoise_tv_chambolle(**kwargs)
    if restore:
        res -= res.min()
        res /= res.max()
        res *= (oldMax-oldMin)
        res += oldMin

    if out is None:
        return res
    else:
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


def guidedFilter(image, epsilon, guidanceImage=None, fMean=None):
    if guidanceImage is None:
        # 1) mean an corr
        meanI = fMean(image)
        meanG = meanI
        corrG  = fMean(image*image)
        corrGI = corrG

        # 2) var and cov    
        varG = corrG - meanG*meanG
        covGI = varG
    else:
        # 1) mean an corr
        meanI = fMean(image)
        meanG = fMean(guidanceImage)
        corrG  = fMean(guidanceImage*guidanceImage)
        corrGI = fMean(guidanceImage*image)

        # 2) var and cov    
        varG = corrG - meanG*meanG
        covGI = corrGI - meanI*meanG

    # 3) "a" and "b" image (as in paper)
    a = covGI / (varG + epsilon)
    b = meanI - a*meanG

    # 4) means on "a" and "b"
    meanA = fMean(a)
    meanB = fMean(b)

    # 5) make result q 
    q = meanA.view(numpy.ndarray)*guidanceImage.view(numpy.ndarray) + meanB.view(numpy.ndarray)

    return q



def gaussianGuidedFilter(image,sigma,epsilon, guidanceImage=None):
    meanFunction = partial(gaussianSmoothing,sigma=sigma)
    return guidedFilter(image, guidanceImage, epsilon, meanFunction)

def medianGuidedFilter(image,radius,epsilon, guidanceImage=None):
    meanFunction = partial(medianSmoothing,radius=radius)
    return guidedFilter(image, guidanceImage, epsilon, meanFunction)


if __name__ == "__main__":
    import numpy
    import vigra

    r=numpy.random.rand(30,30,30).astype(numpy.float32)
    r=vigra.taggedView(r,'xyz')
    res = medianSmoothing(r,radius=3)