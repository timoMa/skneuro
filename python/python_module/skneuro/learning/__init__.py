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

