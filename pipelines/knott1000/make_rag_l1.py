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






print "make the graph and save it"

print "read segmentation"
seg = vigra.impex.readHDF5(opt['oversegL1'], "data")

gridGraph = graphs.gridGraph(seg.shape[0:3])
print "make rag"
# get region adjacency graph from super-pixel labels
rag = graphs.regionAdjacencyGraph(gridGraph, seg, isDense=False)


print "save rag to file"
rag.writeHDF5(opt['ragL1'],'data')



print "extract sizes and length"
edgeLengths = graphs.getEdgeLengths(rag)
vigra.impex.writeHDF5(edgeLengths, opt['ragL1EdgeSize'], "data")

nodeSizes = graphs.getNodeSizes(rag)
vigra.impex.writeHDF5(nodeSizes, opt['ragL1NodeSize'], "data")
