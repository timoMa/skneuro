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

optJsonFile = "opt.json"
opt = wf.loadJson(optJsonFile)


datasetOpts = [opt['train'],opt['test']]
#datasetOpts = [opt['train']]


datafiles = ["rawData", "oversegL0", "oversegL1", "oversegL1Gt", "semanticP0", "boundaryP1"]
datafiles = ["distTransformPMap1","thinnedDistTransformPMap1"]




class RawDataInput(object):
    def __init__(self, raw):
        self.raw = raw

    def __call__(self):
        return self.raw

    def name(self):
        return "raw"

    def free(self):
        self.raw = None


class HessianEw(object):
    def __init__(self, raw, scale ):

        self.raw = raw
        self.scale = scale

    def __call__(self):

        result = blockF.blockwiseHessianOfGaussianLargestEigenvalues(self.raw,scale=self.scale)
        return result

    def name(self):
        scaleStr = float("{0:.2f}".format(self.scale))
        return "hessian_largest_ew_"+str(scaleStr)

    def free(self):
        pass


class GradMag(object):
    def __init__(self, raw, sigma ):

        self.raw = raw
        self.sigma = sigma

    def __call__(self):

        result = blockF.blockwiseGaussianGradientMagnitude(self.raw,sigma=self.sigma)
        return result

    def name(self):
        sigmaStr = float("{0:.2f}".format(self.sigma))
        return "gradmag_"+str(sigmaStr)

    def free(self):
        pass


class PmapL1(object):
    def __init__(self, h5path):
        self.h5path = h5path
        self.data = None

    def __call__(self):

        self.data = vigra.impex.readHDF5(self.h5path,'exported_data')[:,:,:,0]
        return self.data

    def name(self):
        return "pmapl1"

    def free(self):
        self.data = None





for dopt in datasetOpts:

    print "load raw data"
    raw = vigra.impex.readHDF5(dopt['rawData'], dopt['rawDatasetName'])

    print "load rag"
    rag = graphs.loadGridRagHDF5(dopt['ragL1'],'data')
    gridGraph  = rag.baseGraph

    

    print "raw",raw.dtype, raw.shape

    rawFloat32 = raw.astype(numpy.float32)


    featureInputs = [ 
        PmapL1(  dopt['boundaryP1'] ),
        RawDataInput(raw=rawFloat32),
        GradMag(raw=rawFloat32, sigma=1.0),
        GradMag(raw=rawFloat32, sigma=2.0),
        GradMag(raw=rawFloat32, sigma=3.0),
        GradMag(raw=rawFloat32, sigma=4.0),
        GradMag(raw=rawFloat32, sigma=5.0),
        HessianEw(raw=rawFloat32,scale=1.0),
        HessianEw(raw=rawFloat32,scale=2.0),
        HessianEw(raw=rawFloat32,scale=3.0),
        HessianEw(raw=rawFloat32,scale=4.0),
        HessianEw(raw=rawFloat32,scale=5.0),
        HessianEw(raw=rawFloat32,scale=6.0),
        HessianEw(raw=rawFloat32,scale=7.0),
        HessianEw(raw=rawFloat32,scale=9.0),
        HessianEw(raw=rawFloat32,scale=10.0)
    ]


    featureDir = dopt['ragFeatureDir']

    for fi in featureInputs :

        voxelData  = fi()
        name = fi.name()

        print "DO ACCUMULATOION :", name


        with vigra.Timer("accumulate features"):
            eFeatures, nFeatures = learning.accumulateFeatures(rag=rag, volume=voxelData)

        fi.free()

        #with vigra.Timer("save edge features"):
        vigra.impex.writeHDF5(eFeatures, featureDir+name+'.h5', 'edgeFeatures')
        vigra.impex.writeHDF5(nFeatures, featureDir+name+'.h5', 'nodeFeatures')

    #print result.shape, result
