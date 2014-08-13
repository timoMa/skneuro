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


datasetOpts = [opt['train'],opt['test']]
datasetOpts = [opt['train']]


datafiles = ["rawData", "oversegL0", "oversegL1", "oversegL1Gt", "semanticP0", "boundaryP1"]
datafiles = ["distTransformPMap1","thinnedDistTransformPMap1"]

for dopt in datasetOpts:

    print "load raw data"
    raw = vigra.impex.readHDF5(dopt['rawData'], dopt['rawDatasetName'])

    print "load rag"
    rag = graphs.loadGridRagHDF5(dopt['ragL1'],'data')
    gridGraph  = rag.baseGraph

    

    print "compute hessian features"

    