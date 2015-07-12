import numpy
import skneuro
import vigra
from volumina.api import Viewer
from PyQt4.QtGui import QApplication
import skneuro.denoising as dn



p = '/media/tbeier/data/datasets/hhess/2x2x2nm_chunked/data_sub_n_5.h5'
data = vigra.impex.readHDF5(p,'data')[0:200,0:200,0:100].astype('float32').squeeze()




app = QApplication(sys.argv)
v = Viewer()

v.addGrayscaleLayer(data, name="raw")



with vigra.Timer("get ranks 8*2"):
    a = dn.ballRankOrderFilterNew(data,radius=4, rank=0.5,
                                  minVal=0.0, maxVal=255.0)
    
v.addGrayscaleLayer(a, name="0.5 8* 2")




v.setWindowTitle("data")
v.showMaximized()
app.exec_()

