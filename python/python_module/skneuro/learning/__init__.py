from _learning import _accumulateFeatures, AccumulatorOptions




def accumulateFeatures(rag, volume, histMin=None, histMax=None, nBins=100,
                       histSigma=5.0):
    gridGraph = rag.baseGraph
    affiliatedEdges = rag.affiliatedEdges

    if histMin is None:
        histMin = float(volume.min())
    if histMax is None:
        histMax = float(volume.max())

    res = _accumulateFeatures(gridGraph=gridGraph, rag=rag, affiliatedEdges=affiliatedEdges,
                              volume=volume, histMin=float(histMin), histMax=float(histMax),
                              nBins=int(nBins), histSigma=float(histSigma))
    return res
