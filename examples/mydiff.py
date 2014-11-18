import numpy
import skneuro
import vigra
from volumina.api import Viewer
from PyQt4.QtGui import QApplication
import skneuro.blockwise_filters as bf
import skneuro.denoising as dn
import pylab
import h5py

if True:
    #data = vigra.impex.readHDF5('/mnt/CLAWS1/tbeier/data/knott1000/knott-block-full2/d-gt.h5','sbfsem')[0:100,0:100,0:100].astype('float32')
    #dc = data.copy()
    data = vigra.impex.readImage('/home/tbeier/Desktop/10683114_980333491982687_1214393854_o.jpg').astype(numpy.float32)
    data = vigra.resize(data, [data.shape[0]/2, data.shape[1]/2])
    #data = (data[:,:,0]+data[:,:,1]+data[:,:,2])/(3.0)
    dc = data.copy().view(numpy.ndarray)
else :

    if False:

    
        fp = h5py.File("/mnt/CLAWS1/tbeier/data/knott1000_results/pixel_classification/boundary_prob_r1.h5","r")
        pmap = fp['exported_data'][0:300,0:300,0:300,1]

        fd = h5py.File("/mnt/CLAWS1/tbeier/data/knott1000/knott-block-full2/d-gt.h5", "r")
        data = fd['sbfsem'][0:300,0:300,0:300]


        vigra.impex.writeHDF5(data[0:300, 0:300, 0:300], "sub.h5", "data")
        vigra.impex.writeHDF5(pmap[0:300, 0:300, 0:300], "sub.h5", "pmap")
    else :
        d = vigra.impex.readHDF5("sub.h5", "data")[0:300, 0:300, 0:300]
        p = vigra.impex.readHDF5("sub.h5", "pmap")[0:300, 0:300, 0:300]



param = dn.DiffusionParam()
param.strength = 1.0
param.alpha = 0.001
param.maxT = 4.0
param.dt = 0.25
param.useSt = True

param.sigmaSmooth = 0.1
param.sigmaTensor1 = 1.0
param.sigmaTensor2 = 2.5
param.sigmaStep = 0.5
param.C = 1.0
param.m = 1.0


if True:
    if False:
        dc = dn.diffusion2dc(dc, param)
        vigra.impex.writeHDF5(dc, "sub.h5", "dc_0.75_2")
    else :
        dc = vigra.impex.readHDF5("sub.h5", "dc_0.75_2")

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
    if False:
        pmapSmooth2 = dn.diffusion3d(p.copy()*255.0, param)
        vigra.impex.writeHDF5(pmapSmooth2, "sub.h5", "dc_1.00_2.50")
    if True :
        pmapSmooth1 = vigra.impex.readHDF5("sub.h5", "dc_0.75_2")
        pmapSmooth2 = vigra.impex.readHDF5("sub.h5", "dc_1.00_2.50")

    app = QApplication(sys.argv)
    v = Viewer()


    v.addGrayscaleLayer(pmapSmooth1, name="pmapSmooth_0.75_2.00")
    v.addGrayscaleLayer(pmapSmooth2, name="pmapSmooth_1.00_2.50")
    v.addGrayscaleLayer(p, name="pmap")

    v.addGrayscaleLayer(d, name="raw")
    


    #v.addGrayscaleLayer(bf.grayscaleErosion(data,sigma=2.5), name="e")
    #v.addGrayscaleLayer(dt, name="dt")

    v.setWindowTitle("data")
    v.showMaximized()
    app.exec_()
    sys.exit()