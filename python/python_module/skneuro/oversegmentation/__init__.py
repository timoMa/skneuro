from _oversegmentation import *
import vigra
from vigra import numpy
import skneuro.denoising as denoise

def pmapWatershed(pmap, raw, visu=True, seedThreshold=0.6):
    import skneuro
    viewGrayData = [(pmap, "pmap") ]
    viewLabelData= []



    print "densoise"

    pmapD = denoise.tvBregman(pmap, weight=4.5, isotropic=True).astype(numpy.float32)
    pmapG = vigra.filters.gaussianSmoothing(pmap, 1.0)

    viewGrayData.append((pmapG, "pmapGauss"))
    viewGrayData.append((pmapD, "pmapTotalVariation"))
    viewGrayData.append((raw, "raw"))

    #skneuro.addHocViewer(viewGrayData, viewLabelData, visu=visu)

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
    skneuro.addHocViewer(viewGrayData, viewLabelData, visu=visu)

    return seg