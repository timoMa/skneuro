from _utilities import Blocking3d, extractBlock, writeFromBlock

from thread_pool import ThreadPool
import vigra
import numpy
import threading
from sys import stdout
from multiprocessing import cpu_count
import progressbar

def blockwiseCaller(f, margin, blockShape, nThreads, inputKwargs, paramKwagrs, out,
                    verbose=True, printNth=10):
    if nThreads is None:
        nThreads = cpu_count()
    shape = inputKwargs.itervalues().next().shape
    blocking = Blocking3d(shape,blockShape)
    nBlocks = len(blocking)
    pool = ThreadPool(nThreads)


    def threadFunction(f, blocking, blockIndex, margin, inputKwargs, paramKwagrs, out, lock, 
                       doneBlocks=None, printNth=10):

    
        # get the block with border / margin
        block = blocking.blockWithBorder(blockIndex, width=10)

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
    doneBlocks = None
    if verbose:
        doneBlocks = numpy.zeros(nBlocks)
    for blockIndex in range(nBlocks):

        # 2) Add the task to the queue
        pool.add_task(threadFunction, f=f, blocking=blocking, margin=margin,
                      blockIndex=blockIndex, inputKwargs=inputKwargs,
                      paramKwagrs=paramKwagrs, out=out, lock=lock, doneBlocks=doneBlocks)

    pool.wait_completion()
    if verbose:
        stdout.write("\r100.000000 %%")
        stdout.write("\n")
