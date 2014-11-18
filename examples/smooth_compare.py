
import vigra
from vigra import numpy
from time import time

from volumina.api import Viewer
from PyQt4.QtGui import QApplication

import skneuro
from skneuro import denoising






path = "/home/tbeier/Desktop/data.h5"


pnlm  = "/mnt/CLAWS1/tbeier/tmp/pnlm.h5"
pnlm2  = "/mnt/CLAWS1/tbeier/tmp/pnlm2.h5"

# tv bregman
ptvbi2   = "/mnt/CLAWS1/tbeier/tmp/tvbi2.0.h5"
ptvbai20 = "/mnt/CLAWS1/tbeier/tmp/tvbai20.0.h5"

ptvc2  = "/mnt/CLAWS1/tbeier/tmp/tvc2.0.h5"
ptvc5  = "/mnt/CLAWS1/tbeier/tmp/tvc5.0.h5"

pm1  = "/mnt/CLAWS1/tbeier/tmp/median1.h5"
pm2  = "/mnt/CLAWS1/tbeier/tmp/median2.h5"
pm3  = "/mnt/CLAWS1/tbeier/tmp/median3.h5"


pg1  = "/mnt/CLAWS1/tbeier/tmp/gauss1.h5"

pgf5  = "/mnt/CLAWS1/tbeier/tmp/gf5.h5"


pgm3  = "/mnt/CLAWS1/tbeier/tmp/gm3.h5"

if True:
    data = vigra.readHDF5(path, 'data')[0:250,0:255,0:255].astype(numpy.float32)





##########################
# compute non local mean # 
##########################
if False:
    print "non local mean"
    policy = denoising.RatioPolicy(sigma=2.0, meanRatio=0.90, varRatio=0.80)
    res = denoising.nonLocalMean(image=data, policy=policy, patchRadius=2, searchRadius=7, sigmaSpatial=2.0,
                           sigmaPresmoothing=1.0, stepSize=2, iterations=1, verbose=True)
    vigra.impex.writeHDF5(res, pnlm, 'data')

if True:
    print "non local mean"
    policy = denoising.RatioPolicy(sigma=2.0, meanRatio=0.90, varRatio=0.80)
    res = denoising.nonLocalMean(image=data, policy=policy, patchRadius=2, searchRadius=14, sigmaSpatial=1.5,
                           sigmaPresmoothing=1.0, stepSize=2, iterations=1, verbose=True)
    vigra.impex.writeHDF5(res, pnlm2, 'data')

######################
# compute tv bregman #
######################
if False:
    print "tvBregman isotropic"
    res = denoising.tvBregman(data,weight=2.0, isotropic=True)
    vigra.impex.writeHDF5(res, ptvbi2, 'data')
    
    print "tvBregman anisotropic"
    res = denoising.tvBregman(data,weight=20.0, isotropic=False)
    vigra.impex.writeHDF5(res, ptvbai20, 'data')


######################
# compute tv chambolle 
######################
if False:
    print "tvChambolle isotropic"
    res = denoising.tvChambolle(data,weight=0.1)
    vigra.impex.writeHDF5(res, ptvc5, 'data')


######################
# compute median
######################
if False:
    print "median 1"
    res = denoising.medianSmoothing(data,radius=1)
    vigra.impex.writeHDF5(res, pm1, 'data')

    print "median 2"
    res = denoising.medianSmoothing(data,radius=2)
    vigra.impex.writeHDF5(res, pm2, 'data')

    print "median 3"
    res = denoising.medianSmoothing(data,radius=3)
    vigra.impex.writeHDF5(res, pm3, 'data')


######################
# compute median
######################
if False:
    print "gaussian"
    res = denoising.gaussianSmoothing(data,sigma=1.0)
    vigra.impex.writeHDF5(res, pg1, 'data')

######################
# gaussian guided mean
######################
if False:
    print "guided filter"
    res = denoising.gaussianGuidedFilter(image=data, guidanceImage=data,
                                         sigma=2.0, epsilon=30**2)
    vigra.impex.writeHDF5(res, pgf5, 'data')
    
######################
# median guided mean
######################
if False:
    print "guided filter"
    res = denoising.medianGuidedFilter(image=data, guidanceImage=data,
                                         radius=2, epsilon=30**2)
    vigra.impex.writeHDF5(res, pgm3, 'data')
    





if True:

    print "load data"
    data  = vigra.readHDF5(path, 'data')[0:250,0:255,0:255].astype(numpy.float32)

    nlm  = vigra.readHDF5(pnlm, 'data')
    nlm2  = vigra.readHDF5(pnlm2, 'data')

    tvbi2  = vigra.readHDF5(ptvbi2, 'data')
    tvbai20 = vigra.readHDF5(ptvbai20, 'data')
    tvc5 = vigra.readHDF5(ptvc5, 'data')
    m1 = vigra.readHDF5(pm1, 'data')
    m2 = vigra.readHDF5(pm2, 'data')
    m3 = vigra.readHDF5(pm3, 'data')
    g1 = vigra.readHDF5(pg1, 'data')
    gf5 = vigra.readHDF5(pgf5, 'data')
    gm3 = vigra.readHDF5(pgm3, 'data')

    diff = gf5-data 

    app = QApplication(sys.argv)
    v = Viewer()

    print "add layers"
    v.addGrayscaleLayer(data,    name="raw")
    v.addGrayscaleLayer(nlm,   name="non local mean")
    v.addGrayscaleLayer(nlm2,   name="non local mean (on presmoothed)")
    v.addGrayscaleLayer(tvbi2,   name="tv bregman   2.0  isotropic")
    v.addGrayscaleLayer(tvbai20, name="tv bregman   20.0 anisotropic")
    v.addGrayscaleLayer(tvc5,    name="tv Chambolle 2.0")
    v.addGrayscaleLayer(m1,    name="median 1")
    v.addGrayscaleLayer(m2,    name="median 2")
    v.addGrayscaleLayer(m3,    name="median 3")
    v.addGrayscaleLayer(g1.view(numpy.ndarray),    name="gauss 1")
    v.addGrayscaleLayer(gf5.view(numpy.ndarray),    name="gauss guided filter 5")
    v.addGrayscaleLayer(gm3.view(numpy.ndarray),    name="median guided filter 3")

    v.setWindowTitle("smoothings")
    v.showMaximized()
    app.exec_()