import h5py
from .. import Blocking

def defaultChunks(shape=None):
    c = (100,100,100)
    if shape is None:
        return c
    else:
        return tuple(map(min, zip(shape, c)))


def stackRawAndPmapNaive(raw, pmap, out):
    shape = raw.shape[0:3]
    blockShape = defaultChunks(shape)






if __name__ == "__main__":
    
    pass