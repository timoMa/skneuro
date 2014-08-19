import vigra
import vigra.graphs as graphs
import skneuro
import skneuro.blockwise_filters as blockF
import skneuro.learning as learning
import numpy
import gc
import sys
import h5py
from skneuro import workflows as wf
from skneuro import learning as learn
import os


optJsonFile = "opt.json"
opt = wf.loadJson(optJsonFile)


datasetOpts = [opt['train'],opt['test']]

featPerFile = 22


for dopt in datasetOpts:

    featureDir = dopt['ragFeatureDir']
    file_names = [ featureDir+fn for fn in os.listdir(featureDir) if fn.endswith('h5') ]
    sortedFiles = sorted(file_names)
    nFiles = len(sortedFiles)

    print nFiles 



    print "load rag"
    rag = graphs.loadGridRagHDF5(dopt['ragL1'],'data')
    gridGraph  = rag.baseGraph

    nEdges = rag.edgeNum
    maxNodeId = rag.maxNodeId


    eShape = [nEdges,      nFiles, featPerFile]
    nShape = [maxNodeId+1, nFiles, featPerFile]


    allEFeat = numpy.zeros(eShape, dtype=numpy.float32)
    allNFeat = numpy.zeros(nShape, dtype=numpy.float32)

    for i,f in enumerate(sortedFiles):

        print "load",f
        ef = vigra.impex.readHDF5(f, 'edgeFeatures')
        nf = vigra.impex.readHDF5(f, 'nodeFeatures')

        print ef.shape,nf.shape, allEFeat.shape, allNFeat.shape

        allEFeat[:, i, :] = ef[:, :]
        allNFeat[:, i, :] = nf[:, :]

    print "write file"
    vigra.impex.writeHDF5(allEFeat, dopt['ragFeatures'], 'edgeFeatures')
    vigra.impex.writeHDF5(allNFeat, dopt['ragFeatures'], 'nodeFeatures')