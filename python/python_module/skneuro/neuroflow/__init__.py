import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy


class MyDataTreeWidget(pg.DataTreeWidget):
    def __init__(self, *args, **kwargs):
        pg.DataTreeWidget.__init__(self,*args,**kwargs)

class BoxLayout(QtGui.QWidget):

    def __init__(self, parent=None):

        QtGui.QWidget.__init__(self, parent)

        self._setUpUi()
  

        



        #self.resize(300, 150)
    def _setUpUi(self):
        self.widgets=dict()
        self.setWindowTitle('NeuroFlow')
        self.rootBox = QtGui.QVBoxLayout()
        self.setLayout(self.rootBox)
        self.mainSplitter = QtGui.QSplitter(QtCore.Qt.Horizontal)
        self.rootBox.addWidget(self.mainSplitter)

        # data tree
        self.dataTree = self._initDataTreeWidget()
        self.mainSplitter.addWidget(self.dataTree)

        # main tab widget
        self.mainTabWidget = self.QTabWidget()



    def _initDataTreeWidget(self):
        d = {
            'list1': [1,2,3,4,5,6, {'nested1': 'aaaaa', 'nested2': 'bbbbb'}, "seven"],
            'dict1': {
                'x': 1,
                'y': 2,
                'z': 'three'
            },
            'array1 (20x20)': numpy.ones((10,10))
        }

        tree1 = MyDataTreeWidget(data=d)
        return tree1

    def _stackToHdf5Widget(self):
        


if __name__ == "__main__":

    from skneuro import neuroflow as nf

    import sys
    app = QtGui.QApplication(sys.argv)
    tooltip = BoxLayout()
    tooltip.show()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()