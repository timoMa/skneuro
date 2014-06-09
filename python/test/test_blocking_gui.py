from skneuro import utilities
from skneuro import denoising
from skneuro import blockwise_filters
import numpy
import sys
import vigra



import numpy as np
import scipy
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import vigra 



path = "/home/tbeier/Desktop/data.h5"

data = vigra.readHDF5(path, 'data')[0:250,0:250,0:250].astype(numpy.float32)
shape = data.shape
blockShape = [100, 100, 100]



#print "blockwiseGaussianSmoothing"
result = blockwise_filters.blockwiseMedianSmoothing(data, 3, nThreads=3, blockShape=blockShape)

print result.shape

result = result[:, :, :, 2]

print "blockwiseGaussianGradientMagnitude"
#result = blockwise_filters.blockwiseGaussianGradientMagnitude(data,1.0,nThreads=3,blockShape=blockShape)
print "done"
print result.shape, result.dtype



app = QtGui.QApplication([])

## Create window with two ImageView widgets
win = QtGui.QMainWindow()
win.resize(800,800)
win.setWindowTitle('pyqtgraph example: DataSlicing')
cw = QtGui.QWidget()
win.setCentralWidget(cw)
l = QtGui.QGridLayout()
cw.setLayout(l)
imv1 = pg.ImageView()
imv2 = pg.ImageView()
l.addWidget(imv1, 0, 0)
l.addWidget(imv2, 1, 0)
win.show()

roi = pg.LineSegmentROI([[10, 64], [120,64]], pen='r')
imv1.addItem(roi)






def update():
    global result, imv1, imv2
    d2 = roi.getArrayRegion(result, imv1.imageItem, axes=(1,2))
    imv2.setImage(d2)
    
roi.sigRegionChanged.connect(update)


## Display the data
imv1.setImage(result)
#imv1.setHistogramRange(-0.01, 0.01)
#imv1.setLevels(-0.003, 0.003)

update()

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
