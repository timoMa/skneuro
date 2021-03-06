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

rfFolder = dopt['activeRfDir'].encode('ASCII')


treeCount = 1000
noise = 0.010

if False :

    print "do active initial learning"
    learner = learning.ActiveGraphLearning(treeCount=treeCount, noise=noise)
    rf0Path = rfFolder + 'rf0.h5'
    rf0Path = rf0Path.encode('ASCII')
    learner.initialTraining(graphData = graphData, eY=eY, rfPath=rf0Path)


if True :

    learner = learning.ActiveGraphLearning(treeCount=treeCount, noise=noise)
    rfNumber = 10

    while(True):

        print "load rf nr:",rfNumber
        rfPath = rfFolder+"rf%d.h5"%rfNumber

        print rfPath 

        rfNumber+=1
        rfPathNew = rfFolder+"rf%d.h5"%rfNumber
        ret = learner.getNewRf(graphData = graphData, eY=eY, rfPath=rfPath, rfPathNew=rfPathNew)
        
        if ret == 'done':
            print "done"
            break

        if rfNumber >=50 :
            break
