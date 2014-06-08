from _utilities import Blocking3d, extractBlock, writeFromBlock
from thread_pool import ThreadPool
import vigra
import numpy

def blockwiseCaller(f, margin, shape, blockShape, nThreads, inputKwargs, paramKwagrs, out):

    blocking = Blocking3d(shape,blockShape)
    nBlocks = len(blocking)
    pool = ThreadPool(nThreads)

    def threadFunction(f, blocking, blockIndex, margin, inputKwargs, paramKwagrs, out, doneBlocks):
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
        doneBlocks[blockIndex] = 1
        print doneBlocks.sum(), "/", nBlocks

    doneBlocks = numpy.zeros(nBlocks)
    for blockIndex in range(nBlocks):

        # 2) Add the task to the queue
        pool.add_task(threadFunction, f=f, blocking=blocking, margin=margin,
                      blockIndex=blockIndex, inputKwargs=inputKwargs,
                      paramKwagrs=paramKwagrs, out=out, doneBlocks=doneBlocks)

    pool.wait_completion()
