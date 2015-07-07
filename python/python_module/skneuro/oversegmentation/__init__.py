from _oversegmentation import *
import vigra
from vigra import numpy
import skneuro.denoising as denoise
from  skneuro import addHocViewer
from vigra import blockwise as vbw

import skneuro
from scipy.spatial.distance import cdist











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






def largeSeedWatershed(raw, pmap, seeds, membraneWidth = 7.0, visu=False):

    blockShape = (100, )*3
    cOpts = vbw.convOpts

    pmap = numpy.require(pmap, dtype='float32')

    with  vigra.Timer("smoothed tie breaker"):
        # compute a smoothed map as tie breaker
        opts = vbw.convOpts(blockShape=blockShape, sigma=membraneWidth/2.0)
        gaussianSmoothedPmap  = vbw.gaussianSmooth(pmap, options=opts)
        addEps = 0.3
        growingMap = gaussianSmoothedPmap
        growingMap *= addEps

        # get the actual growing map
        growingMap += pmap
        growingMap /= 1.0 + addEps

    with  vigra.Timer("watershedsNew"):
        # do the actual watershed
        growingMap = vigra.taggedView(growingMap, 'xyz')
        seeds = numpy.require(seeds, dtype='uint32')
        seeds = vigra.taggedView(seeds, 'xyz')
        seg,nSeg = vigra.analysis.watershedsNew(image=growingMap, seeds=seeds)

    if visu:
        grayData = [
        (raw, "raw"),
        (pmap,"pmap"),
        (growingMap,"growingMap")
        ]
        segData  = [
            (seeds, "seeds"),
            (seg, "seg")
        ]
        skneuro.addHocViewer(grayData, segData)

    return seg,nSeg


def getLargeSeeds(raw, pmap, membraneWidth , threshold , rank, visu=False):

    with  vigra.Timer("ballRankOrderFilter"):
        # make mebrane wider with ballRankOrderFilter
        r = int(membraneWidth*0.35 + 0.5)
        r = max(r, 1)
        widerPmap1 = denoise.ballRankOrderFilter(pmap, radius=r, rank=rank)
        widerPmap = denoise.ballRankOrderFilter(widerPmap1, radius=r, rank=rank)
        widerPmap1 = None

    with  vigra.Timer("normalize"):
        # renormalize
        widerPmap -= widerPmap.min()
        widerPmap /= widerPmap.max()

    with  vigra.Timer("binarize"):
        # binarize
        binaryPmap = numpy.zeros(widerPmap.shape, dtype='uint8')
        binaryPmap[pmap>threshold] = 1
        #widerPmap = None  # save mem
        if visu == False:
            widerPmap = None


    with  vigra.Timer("multiBinaryDilation"):
        # morphology
        # 1) make membrane wider by  r
        r = int(membraneWidth*1.2 + 0.5)
        r = max(r, 2)
        mBinaryPmapA = vigra.filters.multiBinaryDilation(binaryPmap,r)
        #binaryPmap = None  # save mem
        if visu == False:
            binaryPmap = None

    #with  vigra.Timer("multiBinaryErosion"):
    #    # morphology
    #    # 1) make membrane smaller by  r
    #    r = int(membraneWidth*0.1 + 0.5)
    #    r = max(r, 1)
    #    mBinaryPmapB = vigra.filters.multiBinaryErosion(mBinaryPmapA,r)
    #    if visu == False:
    #        mBinaryPmapA = None

    with  vigra.Timer("labelVolumeWithBackground"):
        # get seeds
        invertedBinaryPmap = 1- mBinaryPmapA
        if visu == False:
            mBinaryPmapB = None

        invertedBinaryPmap = numpy.require(invertedBinaryPmap, dtype='uint32')
        ccImg = vigra.analysis.labelVolumeWithBackground(invertedBinaryPmap)

        if visu == False:
            invertedBinaryPmap = None

    if visu:
        grayData = [
            (raw, "raw"),
            (pmap,"pmap"),
            (widerPmap,"widerPmap"),
            (binaryPmap,"binaryPmap"),
            (mBinaryPmapA,"mBinaryPmapA"),
            #(mBinaryPmapB,"mBinaryPmapB"),
        ]
        segData  = [
            (ccImg, "seeds"),
            #(seg, "seg")
        ]
        skneuro.addHocViewer(grayData, segData)



    return ccImg

def pmapToSegmentation(raw, pmap, membraneWidth, threshold , rank, seeds=None, visu=False):

    if seeds is None:
        # get the (very large) seeds
        seeds = getLargeSeeds(raw=raw, pmap=pmap,
                              membraneWidth=membraneWidth,
                              threshold=threshold, rank=rank,visu=visu)

    # filter seeds?!!

    # growing
    seg,nSeg = largeSeedWatershed(raw=raw, pmap=pmap, seeds=seeds, membraneWidth=membraneWidth,visu=visu)


    return seg,nSeg,seeds




def waterSlic(image, pmap, seedDist, scaling,
              seedSegmentor,threshold=0.5):
    pass


def honyshed(pmap):
    pass

def findBestSeedCloserThanMembrane(seeds, distances, distanceTrafo, membraneDistance):
    """ finds the best seed of the given seeds, that is the seed with the highest value distance transformation."""
    closeSeeds = distances <= membraneDistance
    numpy.zeros_like(closeSeeds)
    # iterate over all close seeds
    maximumDistance = -numpy.inf
    mostCentralSeed = None
    for seed in seeds[closeSeeds]:
        if distanceTrafo[seed[0], seed[1], seed[2]] > maximumDistance:
            maximumDistance = distanceTrafo[seed[0], seed[1], seed[2]]
            mostCentralSeed = seed
    return mostCentralSeed


def nonMaximumSuppressionSeeds(seeds, distanceTrafo):
    """ removes all seeds that have a neigbour that is closer than the the next membrane

    seeds is a list of all seeds, distanceTrafo is array-like
    return is a list of all seeds that are relevant.

    works only for 3d
    """
    seedsCleaned = set()

    # calculate the distances from each seed to the next seeds.
    distances = cdist(seeds, seeds)
    for i in numpy.arange(len(seeds)):
        membraneDistance = distanceTrafo[seeds[i,0], seeds[i,1], seeds[i,2]]
        bestAlternative = findBestSeedCloserThanMembrane(seeds, distances[i,:], distanceTrafo, membraneDistance)
        seedsCleaned.add(tuple(bestAlternative))
    return numpy.array(list(seedsCleaned))


def volumeToListOfPoints(seedsVolume, threshold=0.):
    return numpy.array(numpy.where(seedsVolume > threshold)).transpose()


def placePointsInVolumen(points, shape):
    volumen = numpy.zeros(shape)
    points = numpy.maximum(points, numpy.array((0, 0, 0)))
    points = numpy.minimum(points, numpy.array(shape) - 1)
    for point in (numpy.floor(points)).astype(int):
        volumen[point[0], point[1], point[2]] = 1
    return volumen


def wsDtSegmentation(pmap, pmin, minMembraneSize, minSegmentSize, sigmaMinima, sigmaWeights, cleanCloseSeeds=True):
    # get the thresholded pmap
    binary = numpy.zeros_like(pmap, dtype=numpy.uint32)
    binary[pmap >= pmin] = 1

    # delete small CCs
    labeled = vigra.analysis.labelVolumeWithBackground(binary)
    skneuro.oversegmentation.sizeFilterSegInplace(labeled, int(numpy.max(labeled)), int(minMembraneSize), checkAtBorder=True)

    # use cleaned binary image as mask
    mask = numpy.zeros_like(binary, dtype = numpy.float32)
    mask[labeled > 0] = 1.

    # perform signed dt on mask
    dt = vigra.filters.distanceTransform3D(mask)
    dtInv = vigra.filters.distanceTransform3D(mask, background=False)
    dtInv[dtInv>0] -= 1
    dtSigned = dt.max() - dt + dtInv

    dtSignedSmoothMinima = vigra.filters.gaussianSmoothing(dtSigned, sigmaMinima)
    dtSignedSmoothWeights = vigra.filters.gaussianSmoothing(dtSigned, sigmaWeights)

    seeds = vigra.analysis.localMinima3D(dtSignedSmoothMinima, neighborhood=26, allowAtBorder=True)

    if cleanCloseSeeds:
        seeds = nonMaximumSuppressionSeeds(volumeToListOfPoints(seeds), dt)
        seeds = placePointsInVolumen(seeds, mask.shape).astype(numpy.uint32)

    seedsLabeled = vigra.analysis.labelVolumeWithBackground(seeds)
    segmentation = vigra.analysis.watershedsNew(dtSignedSmoothWeights, seeds = seedsLabeled, neighborhood=26)[0]

    skneuro.oversegmentation.sizeFilterSegInplace(segmentation, int(numpy.max(segmentation)), int(minSegmentSize), checkAtBorder=True)

    segmentation = vigra.analysis.watershedsNew(dtSignedSmoothWeights, seeds = segmentation, neighborhood=26)[0]

    return segmentation
