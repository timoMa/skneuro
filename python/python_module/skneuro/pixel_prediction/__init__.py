import vigra
from vigra import graphs,numpy
import skneuro.denoising as dn
def pmapSizeFilter(neuroCCPmap, minSize):

    shape = neuroCCPmap.shape
    # do thresholding
    neuroCCBinaryPmap  = numpy.zeros(neuroCCPmap.shape, dtype='uint8')
    neuroCCBinaryPmap[neuroCCPmap>0.5] = 1


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



if __name__ == '__main__':
    import skneuro
    data = numpy.random.rand(100,100,100)
    dataM = dn.ballRankOrderFilter(data.astype('float32'),radius=2, rank=0.4)
    dataM = 1-dataM
    sizeFilterd,dataT = pmapSizeFilter(dataM, minSize=10)


    grayData = [(data, "data"),
                (dataM, "dataM"),
                (dataT, "dataT"),
                (sizeFilterd,"sizeFilterd")
    ]
    segData  = []

    skneuro.addHocViewer(grayData, segData)
