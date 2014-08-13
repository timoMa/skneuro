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





if False:

    print "load"
    gt = vigra.impex.readHDF5(opt["groundTruth"],"volume/data").astype(numpy.uint32)[0, :, :, :, 0]

    print gt.shape



    print "load rag"
    # get region adjacency graph from super-pixel labels
    rag = graphs.loadGridRagHDF5(opt['ragL1'], 'data')
    gridGraph  = rag.baseGraph
    overseg = rag.labels


    print "project gt"
    ragGt, gtQuality = rag.projectBaseGraphGt(baseGraphGt=gt)
    vigra.impex.writeHDF5(ragGt, opt['ragL1Gt'], "data")


if False:
    print "load rag"
    # get region adjacency graph from super-pixel labels
    rag = graphs.loadGridRagHDF5(opt['ragL1'], 'data')
    gridGraph  = rag.baseGraph
    overseg = rag.labels

    ragGt = vigra.impex.readHDF5(opt['ragL1Gt'], "data")

    print "project back"
    projectedRagGt = rag.projectLabelsToGridGraph(ragGt)
    vigra.impex.writeHDF5(projectedRagGt, opt['oversegL1Gt'], "data")

    print projectedRagGt.shape

if True:

    print "computation done"
    gt = vigra.impex.readHDF5(opt["groundTruth"],"volume/data").astype(numpy.uint32)[0, :, :, :, 0]
    raw = vigra.impex.readHDF5(opt['rawData'], 'sbfsem').view(numpy.ndarray)

    print gt.dtype,gt.shape

    oversegGt = vigra.impex.readHDF5(opt['oversegL1Gt'], "data")

    grayData = [(raw, "raw")]
    segData  = [(gt, "gt"),(oversegGt,"projectedGt")]
    skneuro.addHocViewer(grayData, segData)