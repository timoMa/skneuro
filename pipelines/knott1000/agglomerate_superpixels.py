import vigra
import vigra.graphs as graphs
import skneuro
import skneuro.oversegmentation as oseg
import skneuro.blockwise_filters as blockF
import numpy
import gc
import sys
from skneuro import workflows as wf

optJsonFile = "opt.json"
opt = wf.loadJson(optJsonFile)






dset = "exported_data"


print "read raw data"
raw = vigra.impex.readHDF5(opt['rawData'], opt['rawDatasetName']).view(numpy.ndarray)
grayData = [(raw, "raw")]
segData  = []




#if False:
#    print "compute eigenvalues of hessian of gaussian"
#    ew = blockF.blockwiseHessianOfGaussianLargestEigenvalues(raw, 2.0, nThreads=20)
#    ew -= ew.min()
#    ew /= ew.max()
#    grayData.append([ew,"hessian ew"])
#    skneuro.addHocViewer(grayData, segData)
#    vigra.impex.writeHDF5(ew, hessianPath, "data")




# make thinned map

if False:
    print "read pmap"
    pmap = boundaryP1 = vigra.impex.readHDF5(opt['boundaryP1'],'exported_data').view(numpy.ndarray)[:, :, :, 0]
    print "done"
    pmap = numpy.require(pmap, dtype=numpy.float32)


    grayData.append([pmap, "pmap"])
    skneuro.addHocViewer(grayData, segData)


    bpmap = numpy.zeros(pmap.shape, dtype=numpy.float32)
    bpmap[pmap > 0.5] = 1

    dist = vigra.filters.distanceTransform3D(bpmap, background=False)

    print "write distance transform"
    vigra.impex.writeHDF5(dist, opt['distTransformPMap1'], "data")

    res = 1.0 - (1.0 / (0.05*dist**2+1))

    grayData.append([dist, "dist"])
    grayData.append([res, "thinned"])

    print "write thinned pmap transform"
    vigra.impex.writeHDF5(res, opt['thinnedDistTransformPMap1'], "data")


    skneuro.addHocViewer(grayData, segData)



if False: 
    print "make the graph and save it"

    print "read segmentation"
    seg = vigra.impex.readHDF5(opt['oversegL0'], "data")
    segData.append([seg, "seg"])

    gridGraph = graphs.gridGraph(seg.shape[0:3])
    print "make rag"
    # get region adjacency graph from super-pixel labels
    rag = graphs.regionAdjacencyGraph(gridGraph, seg, isDense=False)


    print "save rag to file"
    rag.writeHDF5(opt['ragL0'],'data')




if False:
    print "accumulate edge weights and save them"

    print "load rag"
    # get region adjacency graph from super-pixel labels
    rag = graphs.loadGridRagHDF5(opt['ragL0'],'data')
    gridGraph  = rag.baseGraph

    print "read dmap"
    tmap = vigra.impex.readHDF5(opt['thinnedDistTransformPMap1'], 'data')
    tmap = numpy.require(tmap, dtype=numpy.float32)
    grayData.append([tmap, "tmap"])

    print "extract"
    ggTmap = graphs.implicitMeanEdgeMap(gridGraph, tmap)
    ewDmap = rag.accumulateEdgeFeatures(ggTmap)

    vigra.impex.writeHDF5(ewDmap, opt['ragEdgeDmap'], "data")

    print "read pmap"
    pmap = boundaryP1 = vigra.impex.readHDF5(opt['boundaryP1'],'exported_data').view(numpy.ndarray)[:, :, :, 0]
    pmap = numpy.require(pmap, dtype=numpy.float32)

    grayData.append([pmap, "pmap"])


    print "extract"
    ggPmap = graphs.implicitMeanEdgeMap(gridGraph, pmap)
    ewPmap = rag.accumulateEdgeFeatures(ggPmap)
    vigra.impex.writeHDF5(ewPmap, opt['ragEdgePmap'], "data")

if False:

    print "load rag"
    # get region adjacency graph from super-pixel labels
    rag = graphs.loadGridRagHDF5(opt['ragL0'],'data')
    gridGraph  = rag.baseGraph

    print "extract sizes and length"
    edgeLengths = graphs.getEdgeLengths(rag)
    vigra.impex.writeHDF5(edgeLengths, opt['ragL0EdgeSize'], "data")

    nodeSizes = graphs.getNodeSizes(rag)
    vigra.impex.writeHDF5(nodeSizes, opt['ragL0NodeSize'], "data")


# reduce over-segmentation to 10k superpixels or something like that
if True:
    print "read segmentation"
    seg = vigra.impex.readHDF5(opt['oversegL0'], "data")
    segData.append([seg, "seg"])


    print "load rag"
    # get region adjacency graph from super-pixel labels
    rag = graphs.loadGridRagHDF5(opt['ragL0'],'data')
    gridGraph  = rag.baseGraph


    print "read extracted"
    ewDmap =  vigra.impex.readHDF5(opt['ragL0EdgePmap'], "data")
    ewPmap =  vigra.impex.readHDF5(opt['ragL0EdgeDmap'], "data")

   

    print "read extracted sizes"
    nodeSizes = vigra.impex.readHDF5(opt['ragL0NodeSize'], "data")
    edgeLengths = vigra.impex.readHDF5(opt['ragL0EdgeSize'], "data")

    print ""

    print "do agglomerative clustering"
    # do agglomerativeClustering
    print rag

    settings = [
        #(100, 1.0, 0.25),
        #(100, 1.0, 0.50),
        #(100, 1.0, 0.70),
        (1000, 1.0, 0.25),
        #(1000, 1.0, 0.50),
        #(1000, 1.0, 0.70),
        #(5000, 1.0, 0.25),
        #(5000, 1.0, 0.50),
        #(5000, 1.0, 0.70),

        #(20000, 1.0, 0.25),
        #(20000, 1.0, 0.50),
        #(30000, 1.0, 0.70)

    ]


    for s in settings:
        print s

        edgeWeights = (1.0-s[2])*ewDmap + s[2]*ewPmap


        labelsRag = graphs.agglomerativeClustering(graph=rag, edgeWeights=edgeWeights,
                                                   beta=0.000, nodeFeatures=None,
                                                   nodeNumStop=int(s[0]), wardness=float(s[1]),
                                                   edgeLengths=edgeLengths, nodeSizes=nodeSizes)

        print "project back"
        labels = rag.projectLabelsToGridGraph(labelsRag)
        print "labels.shape", labels.shape
        print "project back done"
        vigra.impex.writeHDF5(labels, opt['oversegL1'], "data")

        #segData.append([rag.labels, "segmentation-L0"])
        name = "segmentation-L1_NODES_%d_W_%f_BETA_%f" % (s[0], s[1], s[2])
        print name
        segData.append([labels.copy(), name])




skneuro.addHocViewer(grayData, segData)







#bla =  array[20:40, 60:80, 60:90]
