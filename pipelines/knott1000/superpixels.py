import vigra
import vigra.graphs as graphs
import skneuro
import skneuro.oversegmentation as oseg
import skneuro.blockwise_filters as blockF
import numpy
import gc
import sys
from skneuro import workflows as wf

optJsonFile = "opt.json"
opt = wf.loadJson(optJsonFile)



print "read raw data"
raw = vigra.impex.readHDF5(opt['rawData'], opt['rawDatasetName']).view(numpy.ndarray)
grayData = [(raw, "raw")]
segData  = []


if False:
    print "read pmap"
    pmap = boundaryP1 = vigra.impex.readHDF5(opt['boundaryP1'],'exported_data').view(numpy.ndarray)[:, :, :, 0]
    print "require type"
    pmap = numpy.require(pmap, dtype=numpy.float32)

    minMap = oseg.prepareMinMap(raw=raw, pmap=pmap, visu=False)
    vigra.impex.writeHDF5(minMap, opt['localMinMap'], "data")
    minMap = numpy.array(minMap)



    restrictedPmap = numpy.zeros(pmap.shape, dtype=numpy.uint8)
    restrictedPmap[numpy.where(pmap < 0.75)] = 1

    minMap *= restrictedPmap
    localMin = vigra.analysis.extendedLocalMinima3D(minMap, neighborhood=26)



    seeds = vigra.analysis.labelVolumeWithBackground(localMin, neighborhood=26)

    growMap = vigra.filters.gaussianSmoothing( pmap, 1.0)


    print seeds.shape, growMap.shape

    print "watersheds"
    seg, nseg = vigra.analysis.watersheds(growMap.astype(numpy.float32), seeds=seeds.astype(numpy.uint32))


    print "nseg", nseg

    grayData.append([minMap, "minMap"])
    segData.append([seeds, "seeds"])
    segData.append([seg, "seg"])
    skneuro.addHocViewer(grayData, segData)
    vigra.impex.writeHDF5(seg, opt['oversegL0'], "data")

if True:
    overseg = vigra.impex.readHDF5(opt['oversegL0'], "data")
    segData.append([overseg+1, "oversegL0-"])

    skneuro.addHocViewer(grayData, segData)




#bla =  array[20:40, 60:80, 60:90]
