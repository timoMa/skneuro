import vigra
from vigra import graphs,numpy
import skneuro.denoising as dn











def post_process_semantic_probs(
    rawData,
    probabilities,
    classMeaning = dict(void=0,membrane=1,membraneBoundary=2,mito=3,mitoBoundary=4,stuff=5),
    membraneDiameter = 15.0
    preSmoothPower = 2
):
    assert preSmoothPower >= 1

    if preSmoothPower is not 1:
        probabilities = probabilities**preSmoothPower
        













def computeFeatures(inputFiles, sortedFeatures, featureFolder):


    for inputFile in inputFiles:

        print "processing file ", inputFile







def pmapSizeFilter(neuroCCBinaryPmap, minSize):
    """
    keywords:
        neuroCCBinaryPmap:  zero where membranes are
                            one where neurons are

    """
    shape = neuroCCBinaryPmap.shape


    # get the connected components 
    neuroCCMap  = vigra.analysis.labelVolume(neuroCCBinaryPmap)
    print "min max",neuroCCMap.min(), neuroCCMap.max()
    assert neuroCCMap.min() == 1
    neuroCCMap -= 1

    # get region adjacency graph from super-pixel labels
    gridGraph = graphs.gridGraph(shape[0:3])
    rag = graphs.regionAdjacencyGraph(gridGraph, neuroCCMap)

    # get all sizes
    nodeSizes = rag.nodeSize()


    # nodeColoring
    print "neuroCCBinaryPmap",neuroCCBinaryPmap.dtype,neuroCCBinaryPmap.shape
    nodeColor = rag.accumulateNodeFeatures(neuroCCBinaryPmap.astype('float32'))


    # find to small nodes
    toSmallNodesIds = numpy.where(nodeSizes<minSize)[0]
    toSmallNodesSizes = nodeSizes[toSmallNodesIds]
    toSmallNodesColor = nodeColor[toSmallNodesIds]

    sortedIndices = numpy.argsort(toSmallNodesSizes)
    sortedIndices.shape,sortedIndices
    toSmallNodesIds = toSmallNodesIds[sortedIndices]

    # todo (impl degree map as numpy array)

    for nid in toSmallNodesIds:

        neigbours = []
        for n in rag.neighbourNodeIter(rag.nodeFromId(nid)):
            neigbours.append(n)
        if len(neigbours)==1:
            # flip color
            nodeColor[nid] = int(not bool(nodeColor[nid]))

    pixelColor = rag.projectNodeFeaturesToGridGraph(nodeColor)
    return pixelColor,neuroCCBinaryPmap



class GaussianSmoothing(object):
    def __init__(self, sigmas):
        pass

    def __call___(self, inFile, featureFolder, cachedInput = None):
        pass


if __name__ == '__main__':
    import skneuro
    from skneuro import pixel_prediction as pp

    
    inputFiles = [
        ( ('train.h5', 'data'), 'training'),
        ( ('test.h5', 'data'), 'test'),
    ]

    featureFolder = 'feature_folder'
    sortedFeatures = []
    pp.computeFeatures(inputFiles=inputFiles,
                       sortedFeatures=sortedFeatures,
                       featureFolder=featureFolder)
