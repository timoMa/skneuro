
import vigra
from vigra import numpy
import matplotlib
from matplotlib import pylab
from time import time



import h5py
import numpy

from volumina.api import Viewer
from volumina.pixelpipeline.datasources import LazyflowSource

from lazyflow.graph import Graph
from lazyflow.operators.ioOperators.opStreamingHdf5Reader import OpStreamingHdf5Reader
from lazyflow.operators import OpCompressedCache

from PyQt4.QtGui import QApplication

f = h5py.File("raw.h5", 'w')
d = (255*numpy.random.random((100,200,300))).astype(numpy.uint8)
f.create_dataset("raw", data=d)
f.close()

f = h5py.File("seg.h5", 'w')
d = (10*numpy.random.random((100,200,300))).astype(numpy.uint32)
f.create_dataset("seg", data=d)
f.close()







path = "/home/tbeier/Desktop/data.h5"
rpathA = "/home/tbeier/Desktop/smoothed_10_2_2_4.h5"
rpathB = "/home/tbeier/Desktop/smoothed_10_4_2_4.h5"
rpathC = "/home/tbeier/Desktop/smoothed_5_8_2_2.h5"
rpathD = "/home/tbeier/Desktop/smoothed_2_8_2_1.h5"
rpathE = "/home/tbeier/Desktop/smoothed_1.5_8_2_1_98_70.h5"
rpathF = "/home/tbeier/Desktop/smoothed_1.2_15_2_1_98_70.h5"
rpathG = "/home/tbeier/Desktop/smoothed_1.0_30_2_1_98_75.h5"
rpathH = "/home/tbeier/Desktop/smoothed_1.0_30_2_1_99_80.h5"
#path ="/home/tbeier/Desktop/raw12_13_03.h5"

if True:


    data = vigra.readHDF5(path, 'data')[0:250,0:255,0:255].astype(numpy.float32)
    print data.shape
    #data -= data.min()
    #data /= data.max()
    print "run"
    t0 =time()
    policy = vigra.filters.RatioPolicy(sigma=1.0, meanRatio=0.99, varRatio=0.80)
    #policy = vigra.filters.NormPolicy(sigma=50.0, meanDist=50, varRatio=0.5)
    res = vigra.filters.nonLocalMean3d(data, policy=policy,searchRadius=30,patchRadius=2,nThreads=13,stepSize=2,verbose=True,sigmaMean=1.0)
    t1 = time()
    vigra.impex.writeHDF5(res,rpathH,'data')

if True:   
    data = vigra.readHDF5(path, 'data')[0:250,0:255,0:255].astype(numpy.float32)
    resA = vigra.readHDF5(rpathA, 'data')
    resB = vigra.readHDF5(rpathB, 'data')
    resC = vigra.readHDF5(rpathC, 'data')
    resD = vigra.readHDF5(rpathD, 'data')
    resE = vigra.readHDF5(rpathE, 'data')
    resF = vigra.readHDF5(rpathF, 'data')
    resG = vigra.readHDF5(rpathG, 'data')
    resH = vigra.readHDF5(rpathH, 'data')
    app = QApplication(sys.argv)
    v = Viewer()


    v.addGrayscaleLayer(data, name="raw")
    v.addGrayscaleLayer(resA, name="smoothed 10  2  2 4 -- 95 50")
    v.addGrayscaleLayer(resB, name="smoothed 10  4  2 4 -- 95 50")
    v.addGrayscaleLayer(resC, name="smoothed 5   8  2 2 -- 95 50")
    v.addGrayscaleLayer(resD, name="smoothed 2   8  2 1 -- 95 50")
    v.addGrayscaleLayer(resE, name="smoothed 1.5 8  2 1 -- 98 70")
    v.addGrayscaleLayer(resF, name="smoothed 1.2 15 2 1 -- 98 70")
    v.addGrayscaleLayer(resG, name="smoothed 1.0 30 2 1 -- 98 75")
    v.addGrayscaleLayer(resH, name="smoothed 1.0 30 2 1 -- 99 80")
    v.setWindowTitle("streaming viewer")
    v.showMaximized()
    app.exec_()