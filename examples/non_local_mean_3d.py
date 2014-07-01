
import vigra
from vigra import numpy
import matplotlib
from time import time
import h5py
import skneuro
from skneuro import denoising








path = "/mnt/CLAWS1/tbeier/data/stack_with_holes/data.h5"
resp = "/mnt/CLAWS1/tbeier/data/stack_with_holes/smoothed.h5"

data = vigra.readHDF5(path, 'data')[:,:,:].astype(numpy.float32)

if False:


    print data.shape
    #data -= data.min()
    #data /= data.max()
    print "run"
    t0 =time()
   

    policy = denoising.RatioPolicy(sigma=2.0, meanRatio=0.90, varRatio=0.80)
    res = denoising.nonLocalMean(image=data, policy=policy, patchRadius=1, searchRadius=7, sigmaSpatial=2.0,
                           sigmaPresmoothing=1.0, stepSize=2, iterations=1, verbose=True, nThreads=20)

    t1 = time()
    vigra.impex.writeHDF5(res,resp,'data')

if True:   

    res = vigra.readHDF5(resp, 'data')
    from volumina.api import Viewer
    from PyQt4.QtGui import QApplication
    import numpy
    import vigra




    print "datashape",data.shape
    print "resshape",res.shape

    app = QApplication(sys.argv)
    v = Viewer()


    v.addGrayscaleLayer(data, name="raw")
    v.addGrayscaleLayer(res, name="smoothed")

    v.setWindowTitle("stack with holes")
    v.showMaximized()
    app.exec_()


if True:   

    res = vigra.readHDF5(resp, 'data')
    from volumina.api import Viewer
    from PyQt4.QtGui import QApplication
    import numpy
    import vigra




    print "datashape",data.shape
    print "resshape",res.shape

    app = QApplication(sys.argv)
    v = Viewer()


    v.addGrayscaleLayer(data, name="raw")
    v.addGrayscaleLayer(res, name="smoothed")

    v.setWindowTitle("stack with holes")
    v.showMaximized()
    app.exec_()
