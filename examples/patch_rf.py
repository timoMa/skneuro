import vigra
import numpy
import skneuro
import skneuro.learning as skl

raw = vigra.impex.readHDF5("/mnt/CLAWS1/tbeier/data/knott1000_results/train/d.h5",'sbfsem').astype('float32')[:,:,:,None]
gt  = vigra.impex.readHDF5("/mnt/CLAWS1/tbeier/data/knott1000_results/train/d-gt.h5",'data').astype('uint32')

raw = raw[0:200, 0:200, 0:200]
gt  =  gt[0:200, 0:200, 0:200]

param = skl.PatchRfParam()
prf = skl.PatchRf(param=param)
prf.train(raw, gt)

