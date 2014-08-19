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
edgeCues = vigra.impex.readHDF5(dopt['ragFeatures'], 'edgeFeatures')
nodeCues = vigra.impex.readHDF5(dopt['ragFeatures'], 'nodeFeatures')

print "load sizes"
edgeSize = vigra.impex.readHDF5(dopt['ragL1EdgeSize'], 'data')
nodeSize = vigra.impex.readHDF5(dopt['ragL1NodeSize'], 'data')

print "load label file"
edgeLabels  = vigra.impex.readHDF5(dopt['ragL1EdgeGt'], 'data')


print "load rag"

rag = graphs.loadGridRagHDF5(dopt['ragL1'],'data')
gridGraph  = rag.baseGraph

print "get merge graph"

mg = graphs.mergeGraph(rag)


df = graphs.NeuroDynamicFeatures(rag, mg)


# assign features
df.assignEdgeCues(edgeCues)
df.assignNodeCues(nodeCues)
df.assignEdgeSizes(edgeSize)
df.assignNodeSizes(nodeSize)

# assign labels
df.assignLabels(edgeLabels)

# register callbacks
df.registerCallbacks()




# the training set from level 0
features, labels  = df.computeInitalTrainingSet()

print "Learn the current rf"
df.trainCurrentRf(features, labels)


print features, labels.squeeze()






#print edgeLabels
#
#
#where0 = numpy.where(edgeLabels==0)[0]
#where1 = numpy.where(edgeLabels==1)[0]
#
#f0 = edgeCues[where0,:,:].reshape(-1,22*16)
#f1 = edgeCues[where1,:,:].reshape(-1,22*16)
#
#n0 = len(where0)
#n1 = len(where1)
#
#print n0,n1
#
#n01 = n0+n1
#
#labels   = numpy.zeros(n01, dtype=numpy.uint32)
#features = numpy.zeros((n01,22*16), dtype=numpy.float32)
#labels[n0:n01]=1
#
#
#features[0:n0] = f0[:]
#features[n0:n01] = f1[:]
#
#
#
#rf = vigra.learning.RandomForest(255)
#oob = rf.learnRF(features,labels.reshape([-1,1]))
#
#print oob
#
#
#dynamicFeatures = 