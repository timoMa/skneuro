import vigra
import vigra.graphs as graphs
import skneuro
import skneuro.blockwise_filters as blockF
import numpy
import gc
import sys
import h5py
from skneuro import workflows as wf

optJsonFile = "opt.json"
opt = wf.loadJson(optJsonFile)


datasetOpts = [opt['train'],opt['test']]



# make the region adjacency graph for both datasets
for dopt in datasetOpts:


    raw = vigra.impex.readHDF5(dopt['rawData'], dopt['rawDatasetName']).view(numpy.ndarray)
    grayData = [(raw, "raw")]
    segData  = []

    print "load oversegmentation"
    seg = vigra.impex.readHDF5(dopt['oversegL1'], "data")

    segData.append((seg,'overseg'))
    skneuro.addHocViewer(grayData, segData)


    gridGraph = graphs.gridGraph(seg.shape[0:3])
    print "make rag"
    rag = graphs.regionAdjacencyGraph(gridGraph, seg, isDense=False)

    print "save rag to file"
    rag.writeHDF5(dopt['ragL1'],'data')


    print "extract sizes and length"
    edgeLengths = graphs.getEdgeLengths(rag)
    vigra.impex.writeHDF5(edgeLengths, dopt['ragL1EdgeSize'], "data")

    nodeSizes = graphs.getNodeSizes(rag)
    vigra.impex.writeHDF5(nodeSizes, dopt['ragL1NodeSize'], "data")