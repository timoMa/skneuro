import numpy
import skneuro
import vigra
from volumina.api import Viewer
from PyQt4.QtGui import QApplication
import skneuro.denoising as dn



p = '/home/tbeier/Desktop/blocks/data_sub_3.h5'
data = vigra.impex.readHDF5(p,'data')[0:70,0:70,0:70].astype('float32').squeeze()




app = QApplication(sys.argv)
v = Viewer()

v.addGrayscaleLayer(data, name="raw")


a = dn.ballRankOrderFilter(data,radius=4, rank=0.5)

with vigra.Timer("get ranks 8*2"):
    for i in range(3):
        if i == 0:
            d = data
        dnd = dn.ballRankOrderFilter(d,radius=4, rank=0.5)
        d = dnd
v.addGrayscaleLayer(d, name="0.5 8* 2")




v.setWindowTitle("data")
v.showMaximized()
app.exec_()

