

import vigra
import numpy
import skneuro
import matplotlib
from matplotlib import pylab
from skneuro.clustering import initalized_mini_batch_em
from scipy.cluster.vq import whiten


pathIn     = "/home/tbeier/Desktop/data.h5"
pathHist   = "/home/tbeier/Desktop/hist.h5"
pathHistEw = "/home/tbeier/Desktop/histEw.h5"
pathLabels = "/home/tbeier/Desktop/labels.h5"
pathEw     = "/home/tbeier/Desktop/ew.h5"
pathEwBank = "/home/tbeier/Desktop/ewBank.h5"

pathProb = "/home/tbeier/Desktop/prob.h5"


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


if False:
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




if False:


    rawData  = vigra.impex.readHDF5(pathIn,'data').astype(numpy.float32)
    hist = vigra.impex.readHDF5(pathHistEw,'data').astype(numpy.float32)
    hist2 = vigra.impex.readHDF5(pathHist ,'data').astype(numpy.float32)
    hist3 = vigra.impex.readHDF5( pathEwBank ,'data').astype(numpy.float32)

    hist = numpy.concatenate([hist,hist2,hist3], axis=3)


    mainfeatures = "/home/tbeier/Desktop/main_features.h5"


    vigra.impex.writeHDF5(hist,mainfeatures,'data')

    print "is saved"

if True:
    hist = vigra.impex.readHDF5( "/home/tbeier/Desktop/main_features.h5",'data').astype(numpy.float32)
    hist = hist.squeeze()
    print hist.shape


    batch_size = 10000
    nFeatures = hist.shape[3]
    nClusters = 10
    nIter = 50
    varianceScale  = 10.0
    #X = hist.reshape([nFeatures,-1])

    X = hist.reshape([-1,nFeatures])

    X = whiten(X)






    probs = initalized_mini_batch_em(X=X,nClusters=nClusters,varianceScale=varianceScale, miniBatchSize=batch_size,nInit=10 )







    if False:


        print "preinitalize"
        mbkm=MiniBatchKMeans(n_clusters=nClusters, batch_size=1000, n_init=10)
        mbkm.fit(X)
        centers =  mbkm.cluster_centers_.swapaxes(0,1).astype(numpy.float64)
        X = X.swapaxes(0,1).astype(numpy.float64)

        print "done"



        cAlg = skneuro.clustering.MiniBatchEm(nFeatures=nFeatures,nClusters=nClusters,miniBatchSize=batch_size,nIter=nIter,varScale=varianceScale)

        print "run"
        rindex = numpy.arange(X.shape[1])
        numpy.random.shuffle(rindex)

        #centers = X[:,rindex[0:nClusters]]

        cAlg.initalizeCenters(centers)
        cAlg.run(X) 

        print "pred"
        probs = cAlg.predict(X)

        print "probs.shape",probs.shape

        probs=probs.reshape((nClusters,)+hist.shape[0:3])



    vigra.impex.writeHDF5(probs, pathProb ,'data')


if True:


    rawData  = vigra.impex.readHDF5(pathIn,'data').astype(numpy.float32)
    probs   = vigra.impex.readHDF5(pathProb,'data').astype(numpy.float32)

    nClusters  = probs.shape[0]

    print probs[:,1]

    print "pshape",probs.shape
    print "nClusters",nClusters


    for c in range(nClusters):
        pimg  = probs[c,rawData.shape[0]/2+50, :, :].squeeze()
        raw     = rawData[rawData.shape[0]/2+50, :, :].squeeze()
        f = pylab.figure()
        
        f.add_subplot(1, 2, 0)
        pylab.imshow(pimg, cmap='gray')

        f.add_subplot(1, 2, 1)
        pylab.imshow(raw, cmap ='gray')

        pylab.title('clustering Bin ')
        pylab.show()
        