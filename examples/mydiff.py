import numpy
import skneuro
import vigra
from volumina.api import Viewer
from PyQt4.QtGui import QApplication
import skneuro.blockwise_filters as bf
import skneuro.denoising as dn
import pylab

if False:
    data = vigra.impex.readHDF5('/home/tbeier/knott-block-full2/d.h5','sbfsem')[0:-1,0:-1,500].astype('float32').squeeze()
    vigra.impex.writeHDF5(data,'/home/tbeier/knott-block-full2/slice2d.h5','data')
else :
    #data = vigra.impex.readHDF5('/home/tbeier/knott-block-full2/slice2d.h5','data').astype('float32').squeeze()
    #dc = data.copy()
    data = vigra.impex.readImage('12074.jpg')[:,:,0]
    dc = data.copy().view(numpy.ndarray)


param = dn.DiffusionParam()
param.strength = 1.0
param.alpha = 0.1
param.maxT = 10.0
param.dt = 0.1
param.useSt = True
param.sigmaTensor1 = 1
param.sigmaTensor2 = 2
param.sigmaStep = 0.5
param.C = 1.0
param.m = 1.0
dc = dn.diffusion2d(dc, param)

imgs = [data, dc]

if False:
    for c in range(3):
        dcc = dc[:,:,c]
        dcc-=dcc.min()
        dcc/=dcc.max()

    for c in range(3):
        dcc = data[:,:,c]
        dcc-=dcc.min()
        dcc/=dcc.max()

f = pylab.figure()
for n, arr in enumerate(imgs):
    arr = arr.squeeze()
    f.add_subplot(1, len(imgs), n)
    pylab.imshow(arr.swapaxes(0,1))
pylab.show()
