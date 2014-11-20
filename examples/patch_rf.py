import skneuro
import vigra
import numpy
import skneuro
import skneuro.learning as skl

raw = vigra.impex.readHDF5("/media/tbeier/data/datasets/knott-block/d.h5",'sbfsem').astype('float32')[:,:,:,None]
gt  = vigra.impex.readHDF5("/media/tbeier/data/datasets/knott-block/d-gt.h5",'sbfsem').astype('uint32')

raw = raw[0:60, 0:60, 0:60]
gt  =  gt[0:60, 0:60, 0:60]

param = skl.PatchRfParam()
prf = skl.PatchRf(param=param)
prf.train(raw, gt)
