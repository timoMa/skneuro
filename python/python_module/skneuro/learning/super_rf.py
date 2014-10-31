from progressbar import *               # just a simple progress bar

import pickle


import numpy

from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD, FastICA
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score


class SuperRf(object):

    def __init__(self):
        self.maxIter  = 20
        self.nKPCASamples = 200
        self.r_c      = []
        self.useRawFeatures = True
        self.onlyRaw = False
        self.nClasses = None
        self.transformBatchSize = 90000 


    def fit(self, X,Y,U):
        if self.nKPCASamples == 'sqrt' or self.nKPCASamples >= X.shape[0]:
            self.nKPCASamples = int(numpy.sqrt(X.shape[0])+0.5)
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
                #print 'st',start, stop, nSamples
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
        clf = RandomForestClassifier(n_estimators=20)
        clf.fit(X,Y)
        return clf

    def _getNlMapping(self, X):
        kpca = KernelPCA(n_components=20,kernel="rbf", fit_inverse_transform=False, gamma=0.0004)
        #kpca = FastICA(n_components=20)
        X_pca = kpca.fit_transform(X)
        return X_pca, kpca

    def _getSamples(self, X, Y, U):  


        trainSize = float(self.nKPCASamples)/len(Y)
        trainSize = 0.5
        sss = StratifiedShuffleSplit( y=Y, n_iter=1,train_size=trainSize,
                                     test_size=self.nClasses,random_state=numpy.random.randint(0,10000))

        for useIndex,notUseIndex in sss:
            #print "ui",len(useIndex)
            XX=X[useIndex,:]
            YY=Y[useIndex]
            break
        rUL  = numpy.random.permutation(U.shape[0])[0:100]
        #rUL  = numpy.random.permutation(U.shape[0])[0:self.nKPCASamples]
        UU = U[rUL, :]

        return XX, YY, UU

    def saveToFile(self, filepath):
        pickle.dump([self.r_c,self.nClasses], open( filepath, "wb" ) )
    def loadFromFile(self, filepath):
        self.r_c, self.nClasses = pickle.load(open( filepath, "rb" ))


if True:
    import sklearn
    from sklearn.cross_validation import train_test_split
    digits = load_digits(n_class=10)

    X = sklearn.preprocessing.scale(digits['data'])
    print X.shape
    Y = digits['target']


    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    X_train, X_ul, Y_train, Y_ul = train_test_split(X_train, Y_train, test_size=0.9, random_state=42)

    print "X",X_train.shape
    print "Y",Y_train.shape



   


    rf = SuperRf()

    rf.fit(X_train,Y_train,X_ul)
    P = rf.predict_proba(X_test)
    Y_predicted = numpy.argmax(P,axis=1)
    print Y_predicted[0:30]
    print P.shape
    print Y_test[0:30]
    print "accuracy",accuracy_score(Y_test, Y_predicted)
