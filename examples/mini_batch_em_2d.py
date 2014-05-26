


import vigra
import numpy
import skneuro
import matplotlib
from matplotlib import pylab
import skneuro.clustering
from sklearn.cluster import MiniBatchKMeans
from scipy.cluster.vq import whiten
import sys

path = "12003.jpg"
path = "69015.jpg"
#path = "12074.jpg"
img = vigra.impex.readImage(path)

for c in range(3):
    cimg=img[:,:,c]
    cimg-=cimg.min()
    cimg/=cimg.max()

hist = vigra.histogram.gaussianHistogram(img, minVals=[0.0]*3, maxVals=[1.0]*3,
                                             bins=10, sigma=6.0, sigmaBin=1.5)
hist = hist.reshape(img.shape[0:2]+(-1,))
hSum = hist.sum(axis=2)
hist /= hSum[:, :, numpy.newaxis]

X = hist.reshape([hist.shape[0]*img.shape[1], -1])
X = whiten(X)
#X = X[:,0:2]
nFeatures = X.shape[1]
print "nFeatures" , nFeatures
nClusters = 30
batch_size  = 2000
nIter = 200

print "preinitalize"
mbkm=MiniBatchKMeans(n_clusters=nClusters, batch_size=10000, n_init=10)
mbkm.fit(X)
labels = mbkm.labels_ 

print labels
#sys.exit(1)

centers =  mbkm.cluster_centers_.swapaxes(0,1).astype(numpy.float64)
X = X.swapaxes(0,1).astype(numpy.float64)

print "done"


cAlg = skneuro.clustering.MiniBatchEm(nFeatures=nFeatures,nClusters=nClusters,miniBatchSize=batch_size,nIter=nIter)

print "run"
rindex = numpy.arange(X.shape[1])
numpy.random.shuffle(rindex)

#centers = X[:,rindex[0:nClusters]]

#cAlg.initalizeCenters(X,labels.astype(numpy.uint32) )
cAlg.initalizeCenters(centers)
cAlg.run(X) 

print "pred"
probs = cAlg.predict(X)

print "probs.shape",probs.shape

probs=probs.reshape((nClusters,)+img.shape[0:2])





if True:




    nClusters  = probs.shape[0]

    print probs[:,1]

    print "pshape",probs.shape
    print "nClusters",nClusters


    for c in range(nClusters):
        pimg  = probs[c, :, :].squeeze()
        #raw     = rawData[, :, :].squeeze()
        f = pylab.figure()
        
        f.add_subplot(1, 2, 0)
        pylab.imshow(pimg, cmap='gray')

        #f.add_subplot(1, 2, 1)
        #pylab.imshow(raw, cmap ='gray')

        pylab.title('clustering Bin ')
        pylab.show()
        