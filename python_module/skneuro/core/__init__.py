from _core import *
import h5py
import numpy 

class H5Path(object):
    def __init__(self,filename,dataset):
        self.filename = filename
        self.dataset  = dataset


class Node(object):
    def __init__(self):
        pass


class PixelFeatureHdf5Node(Node):
    def __init__(self,h5path):
        super(PixelFeatureHdf5Node, self).__init__()



class Operator(object):
   def __init__(self):
        pass


class MultiFileFeatureArray(object):
    def __init__(self,files, concatenateAxis=3):
        self.files=files
        self.fileHandles = [None]*len(self.files)
        self.dataHandles = [None]*len(self.files)

        [ h5py.File(path, 'r') for path,dataset in files]

        shape = None
        self.concatenateAxis = concatenateAxis
        for i, (path, dataset) in enumerate(self.files):
            self.fileHandles[i] = h5py.File(path, 'r')
            self.dataHandles[i] = self.fileHandles[i][dataset]
            fileShape =  self.dataHandles[i].shape
            #print fileShape
            if i == 0:
                shape = list(fileShape)
            else:
                for i in range(len(shape)):

                    if i == concatenateAxis:
                        shape[i] += fileShape[i]
                    else:
                        if shape[i] != fileShape[i]:
                            raise RuntimeError("shape mismatch in construction of MultiFileArray")

        self.shape = shape

    def getBlockFeature(self, start, end):
        result = []
        if self.concatenateAxis == 0:
            for data in self.dataHandle:
                print "get block"
                fileBlock = data[:, start[0]:end[0], start[1]:end[1], start[2]:end[2]]
                result.append(fileBlock)
        else:
            for data in self.dataHandles:
                print "get block"
                fileBlock = data[start[0]:end[0], start[1]:end[1], start[2]:end[2], :]
                result.append(fileBlock)

        return numpy.concatenate(result, axis=self.concatenateAxis)

    def getPixelFeature(self, x, y, z):
        if concatenateAxis == 0 :
            indexing = (slice(0,self.shape[0]),) + (x,y,z)
        else:
            indexing = (x,y,z)+(slice(0,self.shape[-1]),)
        assert isinstance(indexing , tuple)
        res = []
        for dataHandle in self.dataHandles:
            fileRes =  dataHandle[indexing]
            res.append(fileRes)


if __name__ == "__main__":

    import numpy as np
    import h5py
    from time import time
    import skneuro
    import skneuro.core

    if True:
        a = np.random.random(size=(100,100,100,50))
        b = np.random.random(size=(100,100,100,20))
        c = np.random.random(size=(100,100,100,15))

        ha = h5py.File('dataA.h5', 'w')
        ha.create_dataset('data', data=a)

        hb = h5py.File('dataB.h5', 'w')
        hb.create_dataset('data', data=b)

        hc = h5py.File('dataC.h5', 'w')
        hc.create_dataset('data', data=c) 

    files = [ ('dataA.h5','data'),('dataB.h5','data'),('dataC.h5','data')]

    array = skneuro.core.MultiFileFeatureArray(files=files)

    print array.shape

    start = (10, 10, 10)
    end   = (50, 50, 50)

    block = array.getBlockFeature(start,end)

    print "block" , block.shape

    sys.exit(1)


    rawData = H5Path("/home/tbeier/src/vigra/vigranumpy/examples/data.h5",'data')



    rawDataNode = PixelFeatureHdf5Node(h5path=rawData)



    