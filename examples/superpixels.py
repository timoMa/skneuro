import vigra
import vigra.graphs as graphs
import skneuro
import skneuro.oversegmentation as oseg
import skneuro.blockwise_filters as blockF
import numpy
import gc
import sys

rawDataPath = "/mnt/CLAWS1/tbeier/data/graham/raw_resized.h5"
pmapPath = "/mnt/CLAWS1/tbeier/data/graham/ilastik/thorstens_ilastik_edge_map_v3.h5"
pmapPath = "/home/tbeier/Desktop/thorstens_ilastik_edge_map_v3.h5"
pmapPath = "/mnt/CLAWS1/tbeier/data/graham/auto_context_prob_r1.h5"
oversegPath  = "/mnt/CLAWS1/tbeier/data/graham/full/overseg2.h5"
aggloSegPath  = "/mnt/CLAWS1/tbeier/data/graham/full/aggloseg.h5"
hessianPath  = "/mnt/CLAWS1/tbeier/data/graham/full/hessianEw.h5"

dset = "exported_data"


print "read raw Data"
raw = vigra.impex.readHDF5(rawDataPath, "raw_r")#.astype(numpy.float32).squeeze()


grayData = [(raw, "raw")]
segData  = []


if False:
    print "read pmap"
    pmap = vigra.impex.readHDF5(pmapPath, "exported_data2")[:, :, :, 0]
    print "require type"
    pmap = numpy.require(pmap, dtype=numpy.float32)

    minMap = oseg.prepareMinMap(raw=raw, pmap=pmap, visu=False)
    localMin = vigra.analysis.extendedLocalMinima3D(minMap, neighborhood=26)
    seeds = vigra.analysis.labelVolumeWithBackground(localMin, neighborhood=26)

    growMap = vigra.filters.gaussianSmoothing( pmap, 1.0)


    print seeds.shape, growMap.shape

    print "watersheds"
    seg, nseg = vigra.analysis.watersheds(growMap.astype(numpy.float32), seeds=seeds.astype(numpy.uint32))


    print "nseg", nseg

    grayData.append([minMap, "minMap"])
    segData.append([seeds, "seeds"])
    segData.append([seg, "seg"])
    skneuro.addHocViewer(grayData, segData)
    vigra.impex.writeHDF5(seg, oversegPath, "data")


if False:
    print "compute eigenvalues of hessian of gaussian"
    ew = blockF.blockwiseHessianOfGaussianLargestEigenvalues(raw, 2.0, nThreads=20)
    ew -= ew.min()
    ew /= ew.max()
    grayData.append([ew,"hessian ew"])
    skneuro.addHocViewer(grayData, segData)
    vigra.impex.writeHDF5(ew, hessianPath, "data")



pmaps = [
    #("normal",'exported_data2',      "/home/tbeier/Desktop/thorstens_ilastik_edge_map_v3.h5"),
    ("autocontext",'data', "/mnt/CLAWS1/tbeier/data/graham/auto_context_prob_r1.h5")
]





# reduce over-segmentation to 10k superpixels or something like that
if True:
    print "read segmentation"
    seg = vigra.impex.readHDF5(oversegPath, "data")
    segData.append([seg, "seg"])

    print "make grid graph"
    shape = seg.shape[0:3]
    gridGraph = graphs.gridGraph(shape[0:3])

    print "make rag"
    # get region adjacency graph from super-pixel labels
    rag = graphs.regionAdjacencyGraph(gridGraph, seg, isDense=False)


    for name, dsetname, pmapPath in pmaps:


        print "read pmap"
        pmap = vigra.impex.readHDF5(pmapPath, dsetname)[:, :, :, 0]
        pmap = numpy.require(pmap, dtype=numpy.float32)

        grayData.append([pmap, "pmap"])


        print "extract"
        ggPmap = graphs.implicitMeanEdgeMap(gridGraph, pmap)
        edgeWeights = rag.accumulateEdgeFeatures(ggPmap)


        print "do agglomerative clustering"
        # do agglomerativeClustering
        labelsRag = graphs.agglomerativeClustering(graph=rag, edgeWeights=edgeWeights,
                                                beta=0.005, nodeFeatures=None,
                                                nodeNumStop=75000, wardness=0.95)

        labels = rag.projectLabelsToGridGraph(labelsRag)
        vigra.impex.writeHDF5(labels, aggloSegPath, "data")


        segData.append([labels+10, "segmentation-"+name])

skneuro.addHocViewer(grayData, segData)







#bla =  array[20:40, 60:80, 60:90]
