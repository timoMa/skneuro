import vigra
import numpy
import skneuro
import matplotlib
from matplotlib import pylab
from scipy.cluster.vq import whiten
from skneuro.clustering import initalized_mini_batch_em

def showResult(raw,probs):
    nClusters  = probs.shape[-1]
    for c in range(nClusters):
        pimg = probs[raw.shape[0]/2,:,:, c].squeeze()
        rimg = raw[raw.shape[0]/2, :, :].squeeze()
        f = pylab.figure()
        
        f.add_subplot(1, 2, 0)
        pylab.imshow(pimg, cmap='gray')

        f.add_subplot(1, 2, 1)
        pylab.imshow(rimg, cmap ='gray')

        pylab.title('clustering Bin ')
        pylab.show()
        



if True:
    hist = vigra.impex.readHDF5( "/home/tbeier/Desktop/main_features.h5",'data').astype(numpy.float32)

    #hist = hist[0:30,0:100,0:100,:]
    #raw = raw[0:30,0:100,0:100]

    nFeatures = hist.shape[3]

    X = hist.reshape( (-1, nFeatures) )
    X = whiten(X)
    print "input shape", hist.shape
    print "X     shape", X.shape

    batch_size = 1000
    nFeatures = hist.shape[3]
    nClusters = 5
    nIter = 200
    varianceScale  = 10.0


    probs = initalized_mini_batch_em(X=X,nClusters=nClusters,varianceScale=varianceScale, miniBatchSize=batch_size,nInit=10 )
    probs = probs.reshape(  hist.shape[0:3] + (-1,) )

    vigra.impex.writeHDF5(probs,"/home/tbeier/Desktop/probs_0.h5",'data')


if True:

    probs = vigra.impex.readHDF5( "/home/tbeier/Desktop/probs_0.h5",'data').astype(numpy.float32)
    raw  = vigra.impex.readHDF5( "/home/tbeier/Desktop/data.h5",'data').astype(numpy.float32)


    showResult(raw,probs)





if True:
    probs = vigra.impex.readHDF5( "/home/tbeier/Desktop/probs_0.h5",'data').astype(numpy.float32)
    nFeatures = probs.shape[-1]
    print nFeatures
    #probs = numpy.array(probs[:,:,:,0:3])
    probs = vigra.taggedView(probs,'xyzc')

    for i in range(probs.shape[-1]):
        cp = probs[:,:,:,i]
        cp-=cp.min()
        cp/=cp.max()

    print probs.shape



    print "compute"
    phist = vigra.histogram.gaussianHistogram(probs, minVals=tuple([0.0]*nFeatures), maxVals=tuple([1.0]*nFeatures),
                                             bins=15, sigma=4.0, sigmaBin=2.0)
    print "save"
    vigra.impex.writeHDF5(phist,"/home/tbeier/Desktop/probs_0_hist.h5",'data')


if True:
    hist = vigra.impex.readHDF5( "/home/tbeier/Desktop/probs_0_hist.h5",'data').astype(numpy.float32)

    #hist = hist[0:30,0:100,0:100,:]
    #raw = raw[0:30,0:100,0:100]

    spatialSize = numpy.prod(hist.shape[0:3])
    nFeatures   = hist.size / spatialSize

    print "size",spatialSize
    print "nFeatures",nFeatures

    X = hist.reshape( (-1, nFeatures) )
    print nFeatures
    print "input shape", hist.shape
    print "X     shape", X.shape


    
    X = whiten(X)

    batch_size = 1000
    nFeatures = X.shape[-1]
    nClusters = 10
    nIter = 200
    varianceScale  = 20.0


    probs = initalized_mini_batch_em(X=X,nClusters=nClusters,varianceScale=varianceScale, miniBatchSize=batch_size,nInit=10 )
    print "probsShape  ",probs.shape 
    probs = probs.reshape(  hist.shape[0:3] + (-1,) )

    print "probsShape",probs.shape 
    assert probs.shape[-1]==nClusters
    vigra.impex.writeHDF5(probs,"/mnt/CLAWS1/tbeier/tmp/probs_1.h5",'data')


if True:

    probs = vigra.impex.readHDF5("/mnt/CLAWS1/tbeier/tmp/probs_1.h5",'data').astype(numpy.float32)
    raw  = vigra.impex.readHDF5( "/home/tbeier/Desktop/data.h5",'data').astype(numpy.float32)

    print probs.shape


    showResult(raw,probs)
