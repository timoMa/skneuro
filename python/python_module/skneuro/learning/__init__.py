import numpy
from _learning import _accumulateFeatures, AccumulatorOptions
from ..parallel import arrayMinMax

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
    def __init__(self):
        pass

    def initalTraining(self, graphData, eY, rfPath, rfPathNew):
        # do the inital training
        mg = graphs.mergeGraph(rag)
        df = graphs.NeuroDynamicFeatures(self.rag, mg)

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
        features, labels  = df.computeInitalTrainingSet()

        print "train random forest"
        rf = vigra.learning.RandomForest(treeCount=100)

        print "save random forest"
        rf.writeHDF5(rfPath, 'rf')

        print "save features and labels "
        vigra.impex.writeHDF5(features, rfPath, 'X')
        vigra.impex.writeHDF5(labels, rfPath, 'Y')




    def getNewRf(graphData, eY, rfPath):

        print "load random forest"
        rf = vigra.learning.RandomForest(rfPath,'rf')

        X = vigra.impex.readHDF5(rfPath, 'X')
        Y = vigra.impex.readHDF5(rfPath, 'Y')

        mg = graphs.mergeGraph(rag)
        df = graphs.NeuroDynamicFeatures(self.rag, mg)

        # assign features
        df.assignEdgeCues(graphData.eX)
        df.assignNodeCues(graphData.nX)
        df.assignEdgeSizes(graphData.eSize)
        df.assignNodeSizes(graphData.nSize)

        # assign labels
        df.assignLabels(eY)

        # register callbacks
        df.registerCallbacks()

        raise RuntimeError("FROM HERE NOT IMPLEMENTED")


        nX, nY = df.getNewFeatures(rf=rf)



        if nX is None or nY is None:
            return "done"


        X = numpy.concatenate([X,aX], axis=0)
        Y = numpy.concatenate([Y,aY], axis=0)

        print "train random forest"
        rf = vigra.learning.RandomForest(treeCount=100)

        print "save random forest"
        rf.writeHDF5(rfPath, 'rf')

        print "save features and labels "
        vigra.impex.writeHDF5(features, rfPath, 'X')
        vigra.impex.writeHDF5(labels, rfPath, 'Y')

        return "not_done"


    def predict(self):
        pass
