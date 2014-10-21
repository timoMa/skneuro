from progressbar import *               # just a simple progress bar

import pickle


import numpy

from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import RandomForestClassifier

class SuperRf(object):

    def __init__(self):
        self.maxIter  = 4
        self.nSamples = 5000
        self.r_c      = []
        self.useRawFeatures = False
        self.onlyRaw = False
        self.nClasses = 2
        self.transformBatchSize = 90000 


    def fit(self, X,Y,U):
        if self.nSamples == 'sqrt':
            self.nSamples = int(numpy.sqrt(X.shape[0])+0.5)
        # get number of classes
        self.nClasses = Y.max()+1








        for i in range(self.maxIter):
            print "train ",i
            # get a subset of the data
            XX,YY,UU = self._getSamples(X, Y, U)
            # concatenate XX and UU
            XXUU = numpy.concatenate([XX,UU], axis=0)
            # get a non linear mapping

            if self.onlyRaw == False:

                XXUU_NL, R = self._getNlMapping(XXUU)
                # extract the labeled part of XXUU_NL
                XX_NL = XXUU_NL[0:XX.shape[0], : ]
                if self.useRawFeatures :
                    XX_NL = numpy.concatenate([XX,XX_NL],axis=1)
            else:
                R = dict()
                XX_NL = XX



            # train a classifier with XX_NL and Y
            
            C = self._trainSubClassifier(XX_NL, YY)

            # store nl mapper R and trained
            # classifier C
            self._store_R_C(R, C)

    def predict_proba(self, U):
        nSamples = U.shape[0]
        classProb =  numpy.zeros([nSamples,self.nClasses])

        tbc = self.transformBatchSize

        for iterNumber,(r,c) in enumerate(self.r_c):
            print iterNumber
            for start in range(0, nSamples, tbc):
                stop = min(start+tbc,nSamples)
                print 'st',start, stop, nSamples
                XX = U[start:stop, :]

                if self.onlyRaw == False:
                    XX_NL = r.transform(XX)
                    if self.useRawFeatures:
                        XX_NL = numpy.concatenate([XX,XX_NL],axis=1)
                else:
                    XX_NL = XX
                P = c.predict_proba(XX_NL)
                CC = classProb[start:stop, :] 
                CC += P[:, :]
        classProb/=len(self.r_c)
        return classProb
    def _store_R_C(self, R, C):
        self.r_c.append((R,C))

    def _trainSubClassifier(self, X, Y):
        clf = RandomForestClassifier(n_estimators=40)
        clf.fit(X,Y)
        return clf

    def _getNlMapping(self, X):
        kpca = KernelPCA(n_components=10,kernel="rbf", fit_inverse_transform=False, gamma=2.5)
        X_pca = kpca.fit_transform(X)
        return X_pca, kpca

    def _getSamples(self, X, Y, U):       

        # get subsets
        w0 = numpy.where(Y==0)[0]
        w1 = numpy.where(Y==1)[0]

        numpy.random.shuffle(w0)
        w0 =w0[0:self.nSamples/2]

        numpy.random.shuffle(w1)
        w1 =w1[0:self.nSamples/2]

        w01 = numpy.concatenate([w0, w1])

        XX = X[w01, :]
        YY = Y[w01]

        rUL  = numpy.random.permutation(U.shape[0])[0:self.nSamples]
        UU = U[rUL, :]

        return XX, YY, UU

    def saveToFile(self, filepath):
        pickle.dump(self.r_c, open( filepath, "wb" ) )
    def loadFromFile(self, filepath):
        self.r_c = pickle.load(open( filepath, "rb" ))


if False:
    NL = 1000*100
    NUL = NL*2
    NF = 20
    X =  numpy.random.random([NL,NF])
    U =  numpy.random.random([NUL,NF])
    Y =  numpy.random.randint(0, 2, NL)

    print X.shape



    rf = SuperRf()

    rf.fit(X,Y,U)
    P = rf.predict_proba(U)

    print 'P',P
