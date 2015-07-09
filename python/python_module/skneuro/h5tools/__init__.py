import h5py
#from .. import Blocking

def defaultChunks(shape=None):
    c = (100,100,100)
    if shape is None:
        return c
    else:
        return tuple(map(min, zip(shape, c)))


def stackRawAndPmapNaive(raw, pmap, out):
    shape = raw.shape[0:3]
    blockShape = defaultChunks(shape)




def pyramideShapes(shape):
    shapes = []
    cshape = shape
    while True:
        b = False
        for s in cshape:
            if s == 1 :
                b = True
                break
        if b :
            break
        cshape = [s/2 for s in cshape]
        shapes.append(cshape)
    print len(shapes)
    return shapes

def volumePyramide(datasetIn, outGroup):
    pass



if __name__ == "__main__":
    
    print pyramideShapes([3300,3300,4000])
