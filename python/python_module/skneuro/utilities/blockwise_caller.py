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
    #print "shape ",shape
    blocking = Blocking3d(shape,blockShape)
    nBlocks = len(blocking)
    #pool = ThreadPool(nThreads)

    #print "nThreads",nThreads
    #print "blockShape",blockShape,blocking.blockShape()
    #print "shape",shape,blocking.shape()
    #print "margin",margin

    def threadFunction(f, blockWithBorder, blockIndex, margin, inputKwargs, paramKwagrs, out, lock, 
                       doneBlocks=None, printNth=10):

        lock.acquire(True)
        #print "border",blockWithBorder.border().begin(),blockWithBorder.border().end()
        #print "core",blockWithBorder.core().begin(),blockWithBorder.core().end()
        #print "localCore",blockWithBorder.localCore().begin(),blockWithBorder.localCore().end()
        lock.release()
        # get the block with border / margin
        #blockWithBorder = blocking.blockWithBorder(blockIndex, width=margin)

        # make the arguments
        kwargs = dict()
        for keyword in inputKwargs.keys():
            # write data from total into block
            array = vigra.taggedView(inputKwargs[keyword], 'xyz')
            blockArray = extractBlock(blockWithBorder, array)
            kwargs[keyword] = blockArray
        kwargs.update(paramKwagrs)

        # do computations
        blockOutput = f(**kwargs).squeeze()
        #write back to global out


        writeFromBlock(blockWithBorder, blockOutput, out)

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
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=nThreads) as executor:

        iterHolder = blocking.blockWithBorderIter([margin]*3)
        for blockIndex in range(nBlocks):
            #print "submit",blockIndex,nBlocks,blockWithBorder
            blockWithBorder = iterHolder.getBlock(blockIndex)
            executor.submit(threadFunction, f=f, blockWithBorder=blockWithBorder, margin=margin,
                            blockIndex=blockIndex, inputKwargs=inputKwargs,
                            paramKwagrs=paramKwagrs, out=out, lock=lock, doneBlocks=doneBlocks)
            blockIndex+=1
            if blockIndex == nBlocks:
                break


        #for blockIndex in range(nBlocks):
        #    executor.submit(threadFunction, f=f, blocking=blocking, margin=margin,
        #                    blockIndex=blockIndex, inputKwargs=inputKwargs,
        #                    paramKwagrs=paramKwagrs, out=out, lock=lock, doneBlocks=doneBlocks)
        del doneBlocks
        doneBlocks = None

        if verbose:
            stdout.write("\r100.000000 %%")
            stdout.write("\n")

    del doneBlocks
    doneBlocks = None