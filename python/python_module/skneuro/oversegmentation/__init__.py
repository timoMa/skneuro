from _oversegmentation import *
import vigra
from vigra import numpy
import skneuro.denoising as denoise
from  skneuro import addHocViewer

def pmapWatershed(pmap, raw, visu=True, seedThreshold=0.6):

    viewGrayData = [(pmap, "pmap") ]
    viewLabelData= []



    print "densoise"

    pmapD = denoise.tvBregman(pmap, weight=4.5, isotropic=True).astype(numpy.float32)
    pmapG = vigra.filters.gaussianSmoothing(pmap, 1.0)

    viewGrayData.append((pmapG, "pmapGauss"))
    viewGrayData.append((pmapD, "pmapTotalVariation"))
    viewGrayData.append((raw, "raw"))

    #addHocViewer(viewGrayData, viewLabelData, visu=visu)

    print "compute local minima "




    localMin = vigra.analysis.extendedLocalMinima3D(pmapD,neighborhood=26)
    localMin2 = localMin.astype(numpy.float32)

    print "tweak min"
    localMin2 *= pmap



    whereZero = numpy.where(localMin == 0)
    localMin2[whereZero] = 100.0

    whereMin = numpy.where(localMin2 <= seedThreshold)

    filteredLocalMin = numpy.zeros(localMin.shape, dtype=numpy.uint8)
    filteredLocalMin[whereMin] = 1

    viewGrayData.append([localMin,"localMin"])
    viewGrayData.append([filteredLocalMin,"filteredLocalMin"])

    # compute connected components
    seeds = vigra.analysis.labelVolumeWithBackground(filteredLocalMin, neighborhood=26)

    viewLabelData.append([seeds, "seeds"])

    print "watersheds"
    seg, nseg = vigra.analysis.watersheds(pmapG.astype(numpy.float32), seeds=seeds.astype(numpy.uint32))

    print "nseg",nseg


    viewLabelData.append([seg, "seg"])
    addHocViewer(viewGrayData, viewLabelData, visu=visu)

    return se




def restrictedLocalMinima(image, minAllowed, neighborhood=26):
    """ discard any min where minAllowed[x,y,y]==0
    """
    localMin = vigra.analysis.extendedLocalMinima3D(numpy.ensure(image,dtype=numpy.float32),neighborhood=neighborhood)
    minNotAllowed = numpy.where(minAllowed==0)
    localMin[minNotAllowed]=0
    return localMin




def prepareMinMap(raw, pmap, sPre=0.8, sInt=5.0, mapInterval=0.5,
                  scaleEw=4.0, ewBeta=0.01,
                  tvWeightSoft=None, isotropicTvSoft=True,
                  tvWeightHard=None, isotropicTvHard=True,
                  sPost=0.6, visu=False
                ):
    """

    """


    print "prepare stuff"
    if tvWeightSoft is None and isotropicTvSoft:
        tvWeightSoft=5.0
    elif tvWeightSoft is None and isotropicTvSoft==False:
        tvWeightSoft=25.0

    if tvWeightHard is None and isotropicTvHard:
        tvWeightHard=0.7
    elif tvWeightHard is None and isotropicTvHard==False:
        tvWeightHard=15.0


    grayData = []
    labelsData = []

    # do minimalistic raw map presmoothing to remove artifacts
    if sPre>0.0001:
        rawG = vigra.filters.gaussianSmoothing(numpy.require(raw ,dtype=numpy.float32), sigma=sPre)
    else :
        rawG = numpy.require(image,dtype=numpy.float32)



    print "pmap integral"
    # get pmap integral
    pmapIntegral = vigra.filters.gaussianSmoothing(numpy.require(pmap, dtype=numpy.float32), sigma=sInt )
    pmapIntegral = numpy.array(pmapIntegral)
    
    grayData.append([rawG,'rawG'])
    grayData.append([pmapIntegral,'pmapIntegral'])

    if visu:
        addHocViewer(grayData, labelsData, visu=visu)

    # remap integral 
    pmapIntegral[pmapIntegral>mapInterval]=mapInterval
    pmapIntegral*=1.0/mapInterval



    print "soft tv"
    # do soft TV smoothing
    pmapTVSoft = denoise.tvBregman(pmap, weight=tvWeightSoft, isotropic=isotropicTvSoft).astype(numpy.float32)


    print "hard tv"
    # do hard heavy TV smoothing
    pmapTVHard = denoise.tvBregman(pmap, weight=tvWeightHard, isotropic=isotropicTvHard).astype(numpy.float32)



    grayData.append([pmapTVSoft,'pmapTVSoft'])
    grayData.append([pmapTVHard,'pmapTVHard'])

    if visu:
        addHocViewer(grayData, labelsData, visu=visu)


    # mix hard and soft according to pmap probability
    mixedPmap = numpy.empty(raw.shape)
    mixedPmap = (1.0 - pmapIntegral)*pmapTVHard  +  pmapIntegral*pmapTVSoft
    

    print "le min le max",mixedPmap.min(), mixedPmap.max()

    #grayData.append([mixedPmap,'mixedPmap'])
    #addHocViewer(grayData, labelsData, visu=visu)

    # add a tiny portion of eigenvalues of hessian give flat wide boundaries the min at the right position
    # but we only add this at places where the boundary is strong (in a hard fashion)
    aew = vigra.filters.hessianOfGaussianEigenvalues(numpy.require(raw, dtype=numpy.float32), scale=scaleEw).squeeze()
    sew = numpy.sort(aew,axis=3)
    ew = sew[:, :, :, 2]
    ew *= pmap**2 
    ew -= ew.min()
    ew /= ew.max()
    ew *= ewBeta

    mixedPmap+=ew


    grayData.append([mixedPmap,'mixedPmapWITHEW'])
    if visu:
        addHocViewer(grayData, labelsData, visu=visu)


    # do minimalistic final smoothing to remove artefacts
    if sPre>0.0001:
        mixedPmapG = vigra.filters.gaussianSmoothing(numpy.require(mixedPmap,dtype=numpy.float32), sigma=sPost)
    else :
        mixedPmapG = numpy.require(mixedPmap,dtype=numpy.float32)

    grayData.append([mixedPmapG,'finalSeedingMap'])
    if visu:
        addHocViewer(grayData, labelsData, visu=visu)


    return mixedPmapG

# IDEA
#
# merge 20 regions, 
# and split 10 regions
#