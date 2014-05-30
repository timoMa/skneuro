from _clustering import *
from scipy.cluster.vq import whiten
from sklearn.cluster import MiniBatchKMeans
import numpy

def initalized_mini_batch_em(X, nClusters, varianceScale, miniBatchSize=1000, nIter=20, nInit=10) :

    nFeatures = X.shape[1]

    print "nfeatures",nFeatures,"nClusters",nClusters


    assert X.shape[0]>X.shape[1]
    mbkm=MiniBatchKMeans(n_clusters=nClusters, batch_size=miniBatchSize, n_init=nInit)

    print "mini batch k means initialization"
    mbkm.fit(X)
    centers =  mbkm.cluster_centers_.swapaxes(0,1).astype(numpy.float64)

    print "swap axes"
    X = X.swapaxes(0,1).astype(numpy.float64)

    print "mini batch em"
    cAlg = MiniBatchEm(nFeatures=nFeatures,nClusters=nClusters,miniBatchSize=miniBatchSize,nIter=nIter,varianceScale=varianceScale)

    print "initialize"
    cAlg.initalizeCenters(centers)
    cAlg.run(X) 

    print "predict"
    probs = cAlg.predict(X)
    probs = probs.swapaxes(0,1)

    return probs