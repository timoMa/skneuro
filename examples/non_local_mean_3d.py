
import vigra
from vigra import numpy
import matplotlib
from time import time
import h5py
import skneuro
from skneuro import denoising
from volumina.api import Viewer
from PyQt4.QtGui import QApplication
import numpy
import vigra










path  = "/mnt/CLAWS1/tbeier/data/stack_with_holes/data.h5"
sPath = "/mnt/CLAWS1/tbeier/data/stack_with_holes/smoothed.h5"
stPath = "/mnt/CLAWS1/tbeier/data/stack_with_holes/smoothedt.h5"
lPath = "/mnt/CLAWS1/tbeier/data/stack_with_holes/labels.h5"
lsPath = "/mnt/CLAWS1/tbeier/data/stack_with_holes/labelsS.h5"
lssPath = "/mnt/CLAWS1/tbeier/data/stack_with_holes/labelsSS.h5"
ewPath    = "/mnt/CLAWS1/tbeier/data/stack_with_holes/ew.h5"
ewsPath    = "/mnt/CLAWS1/tbeier/data/stack_with_holes/ews.h5"
ewssPath    = "/mnt/CLAWS1/tbeier/data/stack_with_holes/ewss.h5"


data = vigra.readHDF5(path, 'data')[0:600,0:600,0:100].astype(numpy.float64)

policy = denoising.RatioPolicy(sigma=1.0, meanRatio=0.95, varRatio=0.8)
nlmp = dict(image=data, policy=policy, patchRadius=2, searchRadius=70, sigmaSpatial=2.0,
            sigmaPresmoothing=1.3, stepSize=2, iterations=1, verbose=True, nThreads=17)

if False:

    print "non truncated"
    smoothed = denoising.nonLocalMean(wTruncate=0.0,**nlmp)
    vigra.impex.writeHDF5(smoothed, sPath, 'data')

if False:

    print "truncated"
    smoothedT = denoising.nonLocalMean(wTruncate=0.15,**nlmp)
    vigra.impex.writeHDF5(smoothedT, stPath, 'data')






if False:
    data=data.astype(numpy.float32)
    ew = vigra.filters.hessianOfGaussianEigenvalues(data, 4.0)
    ew = numpy.sort(ew,axis=3)[:, :, :, 2]
    vigra.impex.writeHDF5(ew, ewPath, 'data')

if False:


    smoothedT = vigra.readHDF5(stPath, 'data').astype(numpy.float32)
    ews = vigra.filters.hessianOfGaussianEigenvalues(smoothedT, 4.0)
    ews = numpy.sort(ews,axis=3)[:, :, :, 2]
    vigra.impex.writeHDF5(ews, ewsPath, 'data')

if False:

    ews = vigra.readHDF5(ewsPath, 'data')

    policy = denoising.RatioPolicy(sigma=1.0, meanRatio=0.95, varRatio=0.8)
    nlmp = dict(image=ews.astype(numpy.float64), policy=policy, patchRadius=2, searchRadius=6, sigmaSpatial=2.0,
                sigmaPresmoothing=1.3, stepSize=2, iterations=1, verbose=True, nThreads=17)



    print "non truncated"
    ewss = denoising.nonLocalMean(wTruncate=0.1,**nlmp)
    vigra.impex.writeHDF5(ewss, ewssPath, 'data')



if False:   

    
    ew = vigra.readHDF5(ewPath, 'data')
    ews = vigra.readHDF5(ewsPath, 'data')


    print "seg "
    labels, nseg = vigra.analysis.watersheds(ew)
    print "seg S"
    labelsS, nseg = vigra.analysis.watersheds(ews)

    vigra.impex.writeHDF5(labels, lPath, 'data')
    vigra.impex.writeHDF5(labelsS, lsPath, 'data')


if False:   


    ewss = vigra.readHDF5(ewssPath, 'data')
    labelsSS, nseg = vigra.analysis.watersheds(ewss.astype(numpy.float32))
    vigra.impex.writeHDF5(labelsSS, lssPath, 'data')




if False :
    print "get percentile filter"
    import scipy
    import scipy.ndimage
    # p percentile 
    ews = vigra.readHDF5(ewsPath, 'data')
    pEws=scipy.ndimage.filters.percentile_filter(ews, percentile=80, size=5)
    vigra.impex.writeHDF5(pEws, "/mnt/CLAWS1/tbeier/data/stack_with_holes/ewssPercentile.h5", 'data')



if True:   


    pEws = vigra.readHDF5("/mnt/CLAWS1/tbeier/data/stack_with_holes/ewssPercentile.h5", 'data')
    labelsPS, nseg = vigra.analysis.watersheds(pEws.astype(numpy.float32))
    vigra.impex.writeHDF5(labelsPS, "/mnt/CLAWS1/tbeier/data/stack_with_holes/lps.h5", 'data')


if True:   

    smoothedT=data.astype(numpy.float32)
    ew = vigra.readHDF5(ewPath, 'data')
    ews = vigra.readHDF5(ewsPath, 'data')
    ewss = vigra.readHDF5(ewssPath, 'data')
    pEws = vigra.readHDF5("/mnt/CLAWS1/tbeier/data/stack_with_holes/ewssPercentile.h5", 'data')
    #dd = ew-ews

    print "seg "
    labels = vigra.readHDF5(lPath, 'data')
    labelsS = vigra.readHDF5(lsPath, 'data')
    #labelsSS = vigra.readHDF5(lssPath, 'data')
    labelsPS = vigra.readHDF5("/mnt/CLAWS1/tbeier/data/stack_with_holes/lps.h5", 'data')

    print ew.shape



    print "datashape",data.shape
    print "resshape",smoothedT.shape

    app = QApplication(sys.argv)
    v = Viewer()

    v.addGrayscaleLayer(data, name="raw")
    v.addGrayscaleLayer(smoothedT, name="smoothedT")
    v.addGrayscaleLayer(ew, name="hessian ew")
    v.addGrayscaleLayer(ews, name="hessian ews")
    v.addGrayscaleLayer(ewss, name="hessian ewss")
    v.addGrayscaleLayer(pEws, name="hessian p80ews")
    #v.addGrayscaleLayer(dd, name="diff ews")
    v.addColorTableLayer(labels, name="labels")
    v.addColorTableLayer(labelsS, name="labelsS")
    #v.addColorTableLayer(labelsSS, name="labelsSS")
    v.addColorTableLayer(labelsPS, name="labelsPS")

    v.setWindowTitle("stack with holes")
    v.showMaximized()
    app.exec_()

