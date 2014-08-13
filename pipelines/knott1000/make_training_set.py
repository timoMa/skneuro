import vigra
import vigra.graphs as graphs
import skneuro
import skneuro.oversegmentation as oseg
import skneuro.blockwise_filters as blockF
import numpy
import gc
import sys
import h5py
from skneuro import workflows as wf

optJsonFile = "opt.json"
opt = wf.loadJson(optJsonFile)


datasetOpts = [opt['train']]


datafiles = ["rawData", "oversegL0", "oversegL1", "oversegL1Gt", "semanticP0", "boundaryP1"]
datafiles = ["distTransformPMap1","thinnedDistTransformPMap1"]

for dopt in datasetOpts:


    print "load rag"
    rag = graphs.loadGridRagHDF5(dopt['ragL1'],'data')
    gridGraph  = rag.baseGraph
    seg = rag.labels

    raw = vigra.impex.readHDF5(dopt['rawData'], dopt['rawDatasetName']).view(numpy.ndarray)
    grayData = [(raw, "raw")]
    segData  = []

    oversegL1Gt = vigra.impex.readHDF5(dopt['oversegL1Gt'], 'data')

    segData.append((oversegL1Gt,'gt'))
    segData.append((seg,'overseg'))
    #skneuro.addHocViewer(grayData, segData)


    print "load  node GT"


    if False:

        print "accumulate",oversegL1Gt.shape
        floatOverseg = oversegL1Gt.astype(numpy.float32)
        ragGt = rag.accumulateNodeFeatures(oversegL1Gt.astype(numpy.float32), acc="min").astype(numpy.uint32)
        print ragGt.shape, rag
        vigra.impex.writeHDF5(ragGt, dopt['ragL1Gt'], 'data')

    print "node gt to edge gt"
    if True:
        nodeGt = vigra.impex.readHDF5( dopt['ragL1Gt'], 'data')
        print nodeGt.dtype
        edgeGt = graphs.nodeGtToEdgeGt(rag, nodeGt, ignoreLabel=0)

        print len(numpy.where(edgeGt == 0)[0]), float(len(numpy.where(edgeGt == 0)[0]))/edgeGt.size
        print len(numpy.where(edgeGt == 1)[0]), float(len(numpy.where(edgeGt == 1)[0]))/edgeGt.size
        print len(numpy.where(edgeGt == 2)[0]), float(len(numpy.where(edgeGt == 2)[0]))/edgeGt.size

        vigra.impex.writeHDF5(edgeGt, dopt['ragL1EdgeGt'], 'data')