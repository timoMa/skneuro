import numpy
from _learning import _accumulateFeatures, AccumulatorOptions
from _learning import *
from ..parallel import arrayMinMax
import vigra
from vigra import graphs 

def accumulatorOptions(select = None, edgeFeatures=True, nodeFeatures=True,
                       sigmaHist = 1.5, nBins = 20, histMin=None, histMax=None):
    if select is None:
        select = ['Mean', 'Variance', 'UserRangeHistogram']
    options = AccumulatorOptions()
    options.select = select
    options.edgeFeatures = edgeFeatures
    options.nodeFeatures = nodeFeatures

    return options



def accumulateFeatures(rag, volume, options=None):

    if options is None :
        options = accumulatorOptions()

    if len(options.histMin) == 1 or len(options.histMax) == 2:
        minVal, maxVal = arrayMinMax(volume)
        print minVal, maxVal
        options.histMin = numpy.array([minVal])
        options.histMax = numpy.array([maxVal])

    gridGraph = rag.baseGraph
    affiliatedEdges = rag.affiliatedEdges
    labels = rag.labels

    res = _accumulateFeatures(gridGraph=gridGraph, rag=rag, labels=labels,
                              affiliatedEdges=affiliatedEdges, volume=volume,
                              options=options)
    return res






class GraphData(object):
    def __init__(self, rag, eX, nX, eSize, nSize):
        self.rag = rag
        self.eX = eX
        self.nX = nX
        self.eSize = eSize
        self.nSize = nSize


class ActiveGraphLearning(object):
    

    def __init__(self, treeCount=1000, noise=0.001):
        self.treeCount = treeCount
        self.noise = noise

    def getFreshRf(self, treeCount=None):
        if treeCount is None:
            treeCount = self.treeCount
        return vigra.learning.RandomForest(treeCount=treeCount, min_split_node_size=4,
                                           sample_classes_individually=True)

    def initialTraining(self, graphData, eY, rfPath):
        # do the inital training
        mg = graphs.mergeGraph(graphData.rag)
        df = graphs.NeuroDynamicFeatures(graphData.rag, mg)

        # assign features
        df.assignEdgeCues(graphData.eX)
        df.assignNodeCues(graphData.nX)
        df.assignEdgeSizes(graphData.eSize)
        df.assignNodeSizes(graphData.nSize)

        # assign labels
        df.assignLabels(eY)

        # register callbacks
        df.registerCallbacks()


        # the training set from level 0
        print "compute inital training set"
        features, labels  = df.computeInitalTrainingSet()

        print "features/labels.shape", features.shape, labels.shape

        print "train random forest"
        rf = self.getFreshRf(treeCount=255)
        oob = rf.learnRF(features, labels)

        print "OOB", oob

        print "save random forest"
        rf.writeHDF5(rfPath, 'rf')

        print "save out of bag"
        oobA = numpy.array([oob],dtype=numpy.float32)
        vigra.impex.writeHDF5(oobA, rfPath, 'oob')

        print "save features and labels "
        vigra.impex.writeHDF5(features, rfPath, 'X')
        vigra.impex.writeHDF5(labels, rfPath, 'Y')






    def getNewRf(self, graphData, eY, rfPath, rfPathNew):

        print "load random forest"
        rf = vigra.learning.RandomForest(rfPath,'rf')

        X = vigra.impex.readHDF5(rfPath, 'X')
        Y = vigra.impex.readHDF5(rfPath, 'Y')

        mg = graphs.mergeGraph(graphData.rag)
        df = graphs.NeuroDynamicFeatures(graphData.rag, mg)

        # assign features
        df.assignEdgeCues(graphData.eX)
        df.assignNodeCues(graphData.nX)
        df.assignEdgeSizes(graphData.eSize)
        df.assignNodeSizes(graphData.nSize)

        # assign labels
        df.assignLabels(eY)

        # register callbacks
        df.registerCallbacks()



        ret = df.getNewFeatureByClustering(rf=rf,noiseMagnitude=self.noise)



        if len(ret) == 1 :
            return "done"

        nN, nX, nY = ret
        print "collected",nN,"new training instances"
        nX = nX[0:nN,:]
        nY = nY[0:nN,:]

        X = numpy.concatenate([X,nX], axis=0)
        Y = numpy.concatenate([Y,nY], axis=0)

        print "train random forest"
        rf = self.getFreshRf(treeCount=1000)

        oob = rf.learnRF(X, Y)

        print oob

        print "save random forest"
        rf.writeHDF5(rfPathNew, 'rf')

        print "save out of bag"
        oobA = numpy.array([oob],dtype=numpy.float32)
        vigra.impex.writeHDF5(oobA, rfPathNew, 'oob')

        print "save features and labels "
        vigra.impex.writeHDF5(X, rfPathNew, 'X')
        vigra.impex.writeHDF5(Y, rfPathNew, 'Y')

        #sys.exit(0)
        return "not_done"


    def predict(self, graphData, rfPath, stopProbs=[0.5], damping=0.05):
        print "load random forest"
        rf = vigra.learning.RandomForest(rfPath,'rf')
        mg = graphs.mergeGraph(graphData.rag)
        df = graphs.NeuroDynamicFeatures(graphData.rag, mg)

        # assign features
        df.assignEdgeCues(graphData.eX)
        df.assignNodeCues(graphData.nX)
        df.assignEdgeSizes(graphData.eSize)
        df.assignNodeSizes(graphData.nSize)

        #NOT assign labels
        #df.assignLabels(eY)

        # register callbacks
        df.registerCallbacks()


        stopProbsArray = numpy.array(stopProbs,dtype=numpy.float32)

        labelsArray = df.predict(rf=rf, stopProbs=stopProbsArray, damping=float(damping))

        rLabels = []
        for i,sp in enumerate(stopProbs):
            print "sp",sp
            rLabels.append(labelsArray[i,:])
        return rLabels
