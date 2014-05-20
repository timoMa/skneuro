import skneuro
import skneuro.clustering
import numpy
import vigra

def test_mini_batch_em():
    nFeatures = 100
    nClusters = 10
    nExamples = 1000
    X = numpy.random.rand(nFeatures,nExamples).astype(numpy.float32)


    cAlg = skneuro.clustering.MiniBatchEm(nFeatures=nFeatures,nClusters=nClusters,miniBatchSize=200,nIter=10)


    print X.shape , X.dtype
    X = vigra.taggedView(X,"xy")
    cAlg.run(X)

    prob = cAlg.predict(X)

    print prob.shape
    assert prob.shape[0] == nClusters
    assert prob.shape[1] == nExamples