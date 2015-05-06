import numpy
import skneuro
import vigra
from volumina.api import Viewer
from PyQt4.QtGui import QApplication
import skneuro.denoising as dn
import pylab

if True:
    data = vigra.impex.readHDF5('/media/tbeier/data/datasets/hhess/2x2x2nm_chunked/data_sub.h5','data')[0:300,0:300,0:300].astype('float32').squeeze()




app = QApplication(sys.argv)
v = Viewer()

v.addGrayscaleLayer(data, name="raw")

a = dn.ballRankOrderFilter(data,radius=7, rank=0.1)

v.addGrayscaleLayer(a, name="0.5")


v.setWindowTitle("data")
v.showMaximized()
app.exec_()

