import skneuro
import vigra
import numpy
import skneuro
import skneuro.learning as skl
import skneuro.blockwise_filters as blockF

#"slicing" : [ [0, 900] , [0, 901], [702, 902] ],

raw = vigra.impex.readHDF5("/home/tbeier/knott-block-full2/d.h5",'sbfsem').astype('float32')
gt  = vigra.impex.readHDF5("/home/tbeier/knott-block-full2/d-gt.h5",'sbfsem').astype('uint32')
#raw = vigra.impex.readHDF5("/media/tbeier/data/datasets/knott-block/d.h5",'sbfsem').astype('float32')
#gt  = vigra.impex.readHDF5("/media/tbeier/data/datasets/knott-block/d-gt.h5",'sbfsem').astype('uint32')

raw = raw[0:150, 0:150, 702:902]
gt = gt[0:150, 0:150, 702:902]

hessian1 = blockF.blockwiseHessianOfGaussianLargestEigenvalues(raw, 1.0, nThreads=10)[:,:,:, None]
#hessian2 = blockF.blockwiseHessianOfGaussianLargestEigenvalues(raw, 2.0, nThreads=10)[:,:,:, None]
#hessian3 = blockF.blockwiseHessianOfGaussianLargestEigenvalues(raw, 3.0, nThreads=10)[:,:,:, None]
#hessian4 = blockF.blockwiseHessianOfGaussianLargestEigenvalues(raw, 4.0, nThreads=10)[:,:,:, None]
#

#features = numpy.concatenate([raw[:,:,:,None], hessian1, hessian2, hessian3, hessian4],axis=3)
features = numpy.concatenate([raw[:,:,:,None], hessian1],axis=3)
#features = numpy.round(features, 2).astype('float32')

print features.shape



param = skl.PatchRfParam()
prf = skl.PatchRf(param=param)
prf.train(features, gt)
