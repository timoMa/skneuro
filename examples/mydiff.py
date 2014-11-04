import numpy
import skneuro
import vigra
from volumina.api import Viewer
from PyQt4.QtGui import QApplication
import skneuro.blockwise_filters as bf
import skneuro.denoising as dn
import pylab

if False:
    data = vigra.impex.readHDF5('/home/tbeier/knott-block-full2/d.h5','sbfsem')[0:200,0:200,200].astype('float32').squeeze()
    vigra.impex.writeHDF5(data,'/home/tbeier/knott-block-full2/slice2d.h5','sbfsem')
else :
    data = vigra.impex.readHDF5('/mnt/CLAWS1/tbeier/data/knott1000/knott-block-full2/d-gt.h5','sbfsem')[0:100,0:100,0:100].astype('float32')
    dc = data.copy()
    data = vigra.impex.readImage('12074.jpg').astype(numpy.float32)
    #data = (data[:,:,0]+data[:,:,1]+data[:,:,2])/(3.0)
    dc = data.copy().view(numpy.ndarray)


param = dn.DiffusionParam()
param.strength =1.0
param.alpha = 0.001
param.maxT = 20.0
param.dt = 0.25
param.useSt = True

param.sigmaSmooth = 0.1
param.sigmaTensor1 = 1.5
param.sigmaTensor2 = 2
param.sigmaStep = 0.5
param.C = 1.0
param.m = 1.0


if True:
    dc = dn.diffusion2dc(dc, param)
    imgs = [data, dc]
    if True:
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

else:
    dc = dn.diffusion3d(dc, param)

    app = QApplication(sys.argv)
    v = Viewer()

    v.addGrayscaleLayer(data, name="raw")
    v.addGrayscaleLayer(dc, name="smoothed")

    #v.addGrayscaleLayer(bf.grayscaleErosion(data,sigma=2.5), name="e")
    #v.addGrayscaleLayer(dt, name="dt")

    v.setWindowTitle("data")
    v.showMaximized()
    app.exec_()

