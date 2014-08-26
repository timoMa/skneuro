from _utilities import Blocking3d, extractBlock, writeFromBlock

import vigra
import numpy
import threading
from sys import stdout
from multiprocessing import cpu_count
import progressbar
import gc
from Queue import Queue
import concurrent.futures

import thread

def blockwiseCaller(f, margin, blockShape, nThreads, inputKwargs, paramKwagrs, out,
                    verbose=True, printNth=10):
    if nThreads is None:
        nThreads = cpu_count()
    shape = inputKwargs.itervalues().next().shape
    blocking = Blocking3d(shape,blockShape)
    nBlocks = len(blocking)
    #pool = ThreadPool(nThreads)


    def threadFunction(f, blocking, blockIndex, margin, inputKwargs, paramKwagrs, out, lock, 
                       doneBlocks=None, printNth=10):

    
        # get the block with border / margin
        block = blocking.blockWithBorder(blockIndex, width=margin)

        # make the arguments
        kwargs = dict()
        for keyword in inputKwargs.keys():
            # write data from total into block
            array = vigra.taggedView(inputKwargs[keyword], 'xyz')
            blockArray = extractBlock(block, array)
            kwargs[keyword] = blockArray
        kwargs.update(paramKwagrs)

        # do computations
        blockOutput = f(**kwargs).squeeze()
        #write back to global out


        writeFromBlock(block, blockOutput, out)

        if doneBlocks is not None:
            doneBlocks[blockIndex] = 1
            if blockIndex % printNth == 0:
                p = doneBlocks.sum()/float(nBlocks)*100.0
                lock.acquire(True)
                #"%.*f" % ( n, f )
                stdout.write("\r%.*f %%" % (2,p))
                stdout.flush()
                lock.release()

    lock = threading.Lock()

    if verbose:
        doneBlocks = numpy.zeros(nBlocks)
    


    del doneBlocks
    doneBlocks = None

    if verbose:
        stdout.write("\r100.000000 %%")
        stdout.write("\n")
