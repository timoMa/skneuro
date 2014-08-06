try:
    hasVolumina=True
    import volumina
    from volumina.api import Viewer
except:
    hasVolumina=False
import sys

from PyQt4.QtGui import QApplication
import numpy
import vigra

class Singleton:
    """
    A non-thread-safe helper class to ease implementing singletons.
    This should be used as a decorator -- not a metaclass -- to the
    class that should be a singleton.

    The decorated class can define one `__init__` function that
    takes only the `self` argument. Other than that, there are
    no restrictions that apply to the decorated class.

    To get the singleton instance, use the `Instance` method. Trying
    to use `__call__` will result in a `TypeError` being raised.

    Limitations: The decorated class cannot be inherited from.

    """

    def __init__(self, decorated):
        self._decorated = decorated

    def Instance(self):
        """
        Returns the singleton instance. Upon its first call, it creates a
        new instance of the decorated class and calls its `__init__` method.
        On all subsequent calls, the already created instance is returned.

        """
        try:
            return self._instance
        except AttributeError:
            self._instance = self._decorated()
            return self._instance

    def __call__(self):
        raise TypeError('Singletons must be accessed through `Instance()`.')

    def __instancecheck__(self, inst):
        return isinstance(inst, self._decorated)


@Singleton
class QApp:
    def __init__(self):
        self.app = QApplication(sys.argv)


def addHocViewer(grayData=None, segData=None, title="viewer",visu=True):
    if visu :
        app = QApp.Instance().app

        v = Viewer()

        if grayData is not None:
            for data, name in grayData:
                if isinstance(data, vigra.arraytypes.VigraArray):
                    v.addGrayscaleLayer(data.view(numpy.ndarray), name=name)
                else:
                    v.addGrayscaleLayer(data, name=name)

        if segData is not None:
            for data, name in segData:
                if isinstance(data, vigra.arraytypes.VigraArray):
                    v.addColorTableLayer(data.view(numpy.ndarray), name=name)
                else:
                    v.addColorTableLayer(data, name=name)

        v.setWindowTitle(title)
        v.showMaximized()
        app.exec_()