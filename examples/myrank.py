import numpy
import skneuro
import vigra
from volumina.api import Viewer
from PyQt4.QtGui import QApplication
import skneuro.blockwise_filters as bf
import skneuro.denoising as dn
import pylab

if True:
    data = vigra.impex.readHDF5('/home/tbeier/knott-block-full2/d.h5','sbfsem')[0:300,0:300,0:300].astype('float32').squeeze()




app = QApplication(sys.argv)
v = Viewer()

v.addGrayscaleLayer(data, name="raw")

a = dn.ballRankOrderFilter(data,radius=5, rank=0.1)
b = dn.ballRankOrderFilter(data,radius=5, rank=0.9)

v.addGrayscaleLayer(a, name="0.1")
v.addGrayscaleLayer(b, name="0.9")
v.addGrayscaleLayer(a*b, name="mult")
v.addGrayscaleLayer((2*a*b)/(a+b), name="harmonic")
v.addGrayscaleLayer(dn.ballRankOrderFilter(data,radius=2, rank=0.1), name="dd")
#v.addGrayscaleLayer(bf.grayscaleErosion(data,sigma=2.5), name="e")
#v.addGrayscaleLayer(dt, name="dt")

v.setWindowTitle("data")
v.showMaximized()
app.exec_()

