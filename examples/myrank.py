import numpy
import skneuro
import vigra
from volumina.api import Viewer
from PyQt4.QtGui import QApplication
import skneuro.denoising as dn



p = '/home/tbeier/Desktop/blocks/data_sub_3.h5'
data = vigra.impex.readHDF5(p,'data')[0:200,:,0:200].astype('uint8').squeeze()




app = QApplication(sys.argv)
v = Viewer()

v.addGrayscaleLayer(data, name="raw")



with vigra.Timer("get ranks 8*2"):
    a = dn.ballRankOrder(data,
        radius=12, 
        takeNth=2,
        ranks=(0.01,0.1, 0.5, 0.9, 0.99),
        useHistogram=True,
        minVal=0.0,
        maxVal=255.0,
        nBins=256)
    
v.addGrayscaleLayer(a, name="0.5 8* 2")




v.setWindowTitle("data")
v.showMaximized()
app.exec_()

