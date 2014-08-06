import numpy
from volumina.api import Viewer
from PyQt4.QtGui import QApplication
import sys


radius = [20]*3
sigma = [3.0, 6.0, 6.0]





def makeFilter(radius,sigma):
        patchSize  = [2*r+1 for r in radius]
        kernel = numpy.zeros(patchSize, dtype=numpy.float32)

        def gauss(x,mean,sigma,gamma=1):
            x=float(x)
            mean=float(mean)
            sigma=float(sigma)
            return  numpy.exp(-0.5*((x-mean)/sigma*gamma**2) ** 2)

        def gauss2d(x,mean,sigma):
            x=float(x)
            mean=float(mean)
            sigma=float(sigma)

            return (-1.0/sigma)*gauss(x,mean,sigma) + (((-x+mean)/sigma)**2)*gauss(x,mean,sigma)


        for x in range(patchSize[0]):
            for y in range(patchSize[1]):
                for z in range(patchSize[2]):

                    dgx = gauss2d(x,radius[0], sigma[0])
                    gy = gauss(y, radius[1], sigma[1])
                    gz = gauss(z, radius[2], sigma[2])
                    kernel[x, y, z] = dgx * gy * gz

        return kernel



kernel  = makeFilter(radius,sigma)
kernel /= numpy.sum(kernel)

kernel -=kernel.min()
kernel/=kernel.max()


app = QApplication(sys.argv)
v = Viewer()



v.addGrayscaleLayer(kernel, name="raw")


v.setWindowTitle("kernel")
v.showMaximized()
app.exec_()