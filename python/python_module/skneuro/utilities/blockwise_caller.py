from _utilities import Blocking3d, extractBlock, writeFromBlock
from thread_pool import ThreadPool


def blockwiseCaller(f, margin, shape, blockShape, nThreads, inputKwargs, paramKwagrs, out):

    blocking = Blocking3d(shape,blockShape)
    nBlocks = len(blocking)
    pool = ThreadPool(nThreads)

    def threadFunction(f, blocking, blockIndex, margin, inputKwargs, paramKwagrs, out):
        print "call threadFunction"
        # get the block with border / margin
        block = blocking.blockWithBorder(blockIndex, width=10)

        # make the arguments
        kwargs = dict()
        for keyword in inputKwargs.keys():
            # write data from total into block
            array = inputKwargs[keyword]
            blockArray = extractBlock(block, array)
            kwargs[keyword] = blockArray
        kwargs.update(paramKwagrs)

        # do computations
        blockOutput = f(**kwargs)

        #write back to global out
        writeFromBlock(block, blockOutput, out)

    for blockIndex in range(nBlocks):

        # 2) Add the task to the queue
        pool.add_task(threadFunction, f=f, blocking=blocking, margin=margin,
                      blockIndex=blockIndex, inputKwargs=inputKwargs,
                      paramKwagrs=paramKwagrs, out=out)

    pool.wait_completion()
