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
dopt = opt['train']

featPerFile = 22

print "load cues"
eX = vigra.impex.readHDF5(dopt['ragFeatures'], 'edgeFeatures')
nX = vigra.impex.readHDF5(dopt['ragFeatures'], 'nodeFeatures')

print "load sizes"
eSize = vigra.impex.readHDF5(dopt['ragL1EdgeSize'], 'data')
nSize = vigra.impex.readHDF5(dopt['ragL1NodeSize'], 'data')

print "load label file"
eY  = vigra.impex.readHDF5(dopt['ragL1EdgeGt'], 'data')


print "load rag"

rag = graphs.loadGridRagHDF5(dopt['ragL1'],'data')



graphData = learning.GraphData(rag=rag, eX=eX, nX=nX, 
                               eSize=eSize, nSize=nSize)

rfFolder = dopt['activeRfDir']

if True :

    learner = learning.ActiveGraphLearning()
    rf0Path = rfFolder + 'rf0.h5'
    learner.initialTraining(graphData = graphData, eY=eY, rfPath=rf0Path)


if True :

    learner = learning.ActiveGraphLearning()
    rfNumber = 0

    while(True):

        print "load rf nr:",rfNumber
        rfPath = "rf%d"%rfNumber
        rfNumber+=1
        rfPathNew = "rf%d"%rfNumber
        ret = learner.getNewRf(graphData = graphData, eY=eY, rfPath=rfPath, rfPathNew=rfPathNew)
        
        if ret == 'done':
            return

