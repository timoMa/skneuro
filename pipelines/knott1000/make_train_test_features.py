import vigra
import vigra.graphs as graphs
import skneuro
import skneuro.blockwise_filters as blockF
import skneuro.learning as learning
import numpy
np = numpy
import gc
import sys
import h5py
from skneuro import workflows as wf
from skneuro import learning as learn

import gc

optJsonFile = "opt.json"
opt = wf.loadJson(optJsonFile)


datasetOpts = [opt['test']]
datasetOpts = [opt['train']]


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
        pass


class HessianEw(object):
    def __init__(self, raw, scale ):

        self.raw = raw
        self.scale = scale

    def __call__(self):

        result = blockF.blockwiseHessianOfGaussianSortedEigenvalues(self.raw,scale=self.scale)
        return result

    def name(self):
        scaleStr = float("{0:.2f}".format(self.scale))
        return "hessian_largest_ew_"+str(scaleStr)

    def free(self):
        pass

class StructTensorEw(object):
    def __init__(self, raw, scale ):

        self.raw = raw
        self.scale = scale

    def __call__(self):

        result = blockF.blockwiseStructureTensorSortedEigenvalues(self.raw,innerScale=self.scale, outerScale=self.scale*2.0)
        return result

    def name(self):
        scaleStr = float("{0:.2f}".format(self.scale))
        return "struct_tensor_largest_ew_"+str(scaleStr)

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


class GaussSmooth(object):
    def __init__(self, raw, sigma ):

        self.raw = raw
        self.sigma = sigma

    def __call__(self):

        result = blockF.blockwiseGaussianSmoothing(self.raw,sigma=self.sigma)
        return result

    def name(self):
        sigmaStr = float("{0:.2f}".format(self.sigma))
        return "gauss_smooth"+str(sigmaStr)

    def free(self):
        pass


class Dilation(object):
    def __init__(self, raw, sigma ,name):

        self.raw = raw
        self.sigma = sigma
        self.nn = name
    def __call__(self):

        result = blockF.blockwiseMultiGrayscaleDilation(self.raw,sigma=self.sigma)
        return result

    def name(self):
        sigmaStr = float("{0:.2f}".format(self.sigma))
        return "diletation_"+self.nn+"_"+str(sigmaStr)

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
        del self.data 



class FromH5File(object):
    def __init__(self, h5path, name):
        self.h5path = h5path
        self.data = None    
        self.nn = name
    def __call__(self):

        self.data = vigra.impex.readHDF5(*self.h5path)
        return self.data

    def name(self):
        return self.nn

    def free(self):
        del self.data 



def fRange(start,stop,n=None, endpoint=True):
    if n is None:
        n = stop - start
        n = int(n+0.1)
    r = np.linspace(start, stop ,n,endpoint=endpoint)
    r = tuple( [float(f) for f in r] )
    return r

for dopt in datasetOpts:

    print "load raw data"
    raw = vigra.impex.readHDF5(dopt['rawData'], dopt['rawDatasetName'])

    print "load rag"
    rag = graphs.loadGridRagHDF5(dopt['ragL1'],'data')
    gridGraph  = rag.baseGraph

    

    print "raw",raw.dtype, raw.shape

    rawFloat32 = raw.astype(numpy.float32)


    featureInputs = [ 
        # raw data itself
        RawDataInput(raw=rawFloat32), 
        # learned pixel maps
        PmapL1(  dopt['boundaryP1'] ),
        FromH5File(  ( dopt['semanticP0'],'data'),name='semanticP0' ),
        FromH5File(  ( dopt['thinnedDistTransformPMap1'],'data'),name='thinnedDistTransformPMap1' ),
        # primitive filters on raw data
        [ GradMag(raw=rawFloat32, sigma=s)          for s in fRange(1.0, 5.0,  5)  ],
        [ GaussSmooth(raw=rawFloat32, sigma=s)      for s in fRange(1.0, 5.0,  5)  ],
        [ HessianEw(raw=rawFloat32, scale=s)        for s in fRange(1.0, 10.0, 10) ],
        [ StructTensorEw(raw=rawFloat32, scale=s)   for s in fRange(1.0, 5.0,  5)  ],
        [ Dilation(raw=rawFloat32, sigma=s, name='on_raw')         for s in fRange(1.0, 5.0,  5)  ]
    ]

    # maybe merge list
    mInputs = []

    for fi in featureInputs:
        if isinstance(fi,list):
            for fii in fi :
                mInputs.append(fii)
        else :
            mInputs.append(fi)


    featureDir = dopt['ragFeatureDir']

    for fi in mInputs :


        name = fi.name()
        voxelData  = fi().squeeze()
        

        if voxelData.ndim == 3:
            voxelData = voxelData[:, :, :, None]


        


        for c in range(voxelData.shape[3]):

            vd = voxelData[:, :, :, c]
            print "DO ACCUMULATOION :", name

            #raise RuntimeError("check for existence here")

            with vigra.Timer("accumulate features"):
                eFeatures, nFeatures = learning.accumulateFeatures(rag=rag, volume=vd)

            #with vigra.Timer("save edge features"):
            vigra.impex.writeHDF5(eFeatures, featureDir+name+'_c_%d.h5'%c, 'edgeFeatures')
            vigra.impex.writeHDF5(nFeatures, featureDir+name+'_c_%d.h5'%c, 'nodeFeatures')

            del eFeatures
            del nFeatures
            del vd

            eFeatures = None
            nFeatures = None
            vd = None

        del voxelData
        voxelData = None
        fi.free()
        gc.collect()   

        del fi 
        fi = None

    #print result.shape, result
