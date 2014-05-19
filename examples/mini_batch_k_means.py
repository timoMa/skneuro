

import vigra
import numpy
import skneuro
import matplotlib
from matplotlib import pylab
import skneuro.clustering
from sklearn.cluster import MiniBatchKMeans\

import sys

pathIn     = "/home/tbeier/Desktop/data.h5"
pathHist   = "/home/tbeier/Desktop/hist.h5"
pathHistEw = "/home/tbeier/Desktop/histEw.h5"
pathLabels = "/home/tbeier/Desktop/labels.h5"
pathEw     = "/home/tbeier/Desktop/ew.h5"
pathEwBank = "/home/tbeier/Desktop/ewBank.h5"

if False:
    rawData  = vigra.impex.readHDF5(pathIn,'data').astype(numpy.float32)
    eigenvals = vigra.filters.hessianOfGaussianEigenvalues(rawData,scale=2.0)

    eigenvals  = numpy.sort(eigenvals, axis=3)
    eigenvals  = eigenvals[:, :, :, 0]
    eigenvals -= eigenvals.min()
    eigenvals /= eigenvals.max()

    ew  = eigenvals[rawData.shape[0]/2, :, :].squeeze()


    raw     = rawData[rawData.shape[0]/2, :, :].squeeze()
    f = pylab.figure()
    
    f.add_subplot(1, 2, 0)
    pylab.imshow(ew, cmap=matplotlib.cm.Greys_r)

    f.add_subplot(1, 2, 1)
    pylab.imshow(raw, cmap=matplotlib.cm.Greys_r)

    pylab.title('hessian ew Bin %i ')
    pylab.show()

    vigra.impex.writeHDF5(eigenvals,pathEw,'data')


if True:
    rawData  = vigra.impex.readHDF5(pathIn,'data').astype(numpy.float32)

    ews = []

    for scale in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
        print scale
        eigenvals = vigra.filters.hessianOfGaussianEigenvalues(rawData,scale=2.0)
        eigenvals  = numpy.sort(eigenvals, axis=3)
        eigenvals  = eigenvals[:, :, :, 0, None]
        ews.append(eigenvals)

    ews = numpy.concatenate(ews, axis=3)


    vigra.impex.writeHDF5(ews, pathEwBank, 'data')

if False:
    rawData  = vigra.impex.readHDF5(pathIn,'data').astype(numpy.float32)
    print rawData.shape


    rawData = rawData.reshape(rawData.shape+(1,))
    rawData -= rawData.min()
    rawData /= rawData.max()
    print "rawData shape", rawData.shape

    hist = vigra.histogram.gaussianHistogram(rawData, minVals=(0.0,), maxVals=(1.0,),
                                             bins=30, sigma=2.0, sigmaBin=2.0)
    hist = hist.squeeze()
    print "Histogram shape", hist.shape
    # normalize
    hSum = hist.sum(axis=3)
    hist /= hSum[:, :, :, numpy.newaxis]
    vigra.impex.writeHDF5(hist,pathHist,'data')

if False:
    print "ew hist"
    ewData  = vigra.impex.readHDF5(pathEw,'data').astype(numpy.float32)

    ewData = ewData.reshape(ewData.shape+(1,))

    ewHist = vigra.histogram.gaussianHistogram(ewData, minVals=(0.0,), maxVals=(1.0,),
                                             bins=30, sigma=2.0, sigmaBin=2.0)
    ewHist = ewHist.squeeze()
    print "Histogram shape", ewHist.shape
    # normalize
    hSum = ewHist.sum(axis=3)
    ewHist /= hSum[:, :, :, numpy.newaxis]
    vigra.impex.writeHDF5(ewHist,pathHistEw,'data')




if True:


    rawData  = vigra.impex.readHDF5(pathIn,'data').astype(numpy.float32)
    #hist  = vigra.impex.readHDF5(pathHistEw,'data').astype(numpy.float32)
    #hist2  = vigra.impex.readHDF5(pathHist ,'data').astype(numpy.float32)

    hist3 = vigra.impex.readHDF5( pathEwBank ,'data').astype(numpy.float32)

    print "hist bank ",hist3.shape

    if False:
        for binIndex in range(rawData.shape[-1]):
            himg  = hist[rawData.shape[0]/2, :, :, binIndex].squeeze()
            dimg  = rawData[rawData.shape[0]/2, :, :].squeeze()
            f = pylab.figure()
            for n, arr in enumerate([dimg, himg]):
                f.add_subplot(1, 2, n)
                pylab.imshow(arr, cmap=matplotlib.cm.Greys_r)

            pylab.title('Gaussian Histogram Bin %i ' % binIndex)
            pylab.show()

    features = hist3.reshape([-1,hist3.shape[-1]])
    #featuresB = hist2.reshape([-1,hist.shape[-1]])

    #features = numpy.concatenate([featuresA,featuresB],axis=1)

    print features.shape

    batch_size = 100
    n_clusters = 10

    print "construct kmeans"
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, batch_size=batch_size,
                      n_init=10, max_no_improvement=10, verbose=0)

    print "fit kmeans"
    mbk.fit(features)
    labels = mbk.labels_
    labels = labels.reshape(hist.shape[0:3])

    print "save labeling"
    vigra.impex.writeHDF5(labels,pathLabels,'data')


if False:


    rawData  = vigra.impex.readHDF5(pathIn,'data').astype(numpy.float32)
    labels   = vigra.impex.readHDF5(pathLabels,'data').astype(numpy.float32)

    if True:
        cmap = matplotlib.colors.ListedColormap ( numpy.random.rand ( 256,3))
        labels  = labels[rawData.shape[0]/2+50, :, :].squeeze()
        raw     = rawData[rawData.shape[0]/2+50, :, :].squeeze()
        f = pylab.figure()
        
        f.add_subplot(1, 2, 0)
        pylab.imshow(labels, cmap=cmap)

        f.add_subplot(1, 2, 1)
        pylab.imshow(raw, cmap=matplotlib.cm.Greys_r)

        pylab.title('clustering Bin %i ')
        pylab.show()