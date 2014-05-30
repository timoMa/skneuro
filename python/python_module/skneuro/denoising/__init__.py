from scipy.ndimage.filters import median_filter
try:
    from skimage.filter import denoise_tv_bregman,denoise_tv_chambolle
except:
    from skimage.restoration import denoise_tv_bregman,denoise_tv_chambolle



def nonLocalMean():
    pass


def gaussianSmoothing():
    pass


def medianSmoothing(image,radius=None, footprint=None, mode='reflect', cval=0.0, origin=0, out=None):
    size = 2*radius+1
    res = median_filter(input=image, size=size, footprint=footprint, output=out, mode=mode, cval=cval, origin=origin)
    if out is None:
        return res
    else:
        out[:] = res[:]
        return res

def tvSmoothing(image, weight, eps=0.001, maxIter=None, isotropic=True, out=None):


    res = denoise_tv_bregman(image=image, weight=weight, eps=eps, maxIter=maxIter, isotropic=isotropic)
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


def guidedFilter(image, guidanceImage, radius, epsilon,
                 fMean=None):
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
    q = meanA*guidanceImage + meanB 

    return q


if __name__ == "__main__":
    import numpy
    import vigra

    r=numpy.random.rand(30,30,30).astype(numpy.float32)
    r=vigra.taggedView(r,'xyz')
    res = medianSmoothing(r,radius=3)