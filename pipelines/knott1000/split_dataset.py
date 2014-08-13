import vigra
import vigra.graphs as graphs
import skneuro
import skneuro.oversegmentation as oseg
import skneuro.blockwise_filters as blockF
import numpy
import gc
import sys
import h5py
from skneuro import workflows as wf

optJsonFile = "opt.json"
opt = wf.loadJson(optJsonFile)


datasetOpts = [opt['train'],opt['test']]

datafiles = ["distTransformPMap1", "thinnedDistTransformPMap1", "rawData", "oversegL0",
             "oversegL1", "oversegL1Gt", "semanticP0", "boundaryP1"]
datafiles = ["oversegL0", "oversegL1"]
for dopt in datasetOpts:
    slicing = dopt['slicing']
    for df in datafiles :
        print "FILE:", df

        print "    load:"

        fullFile = opt[df]
        subFile  = dopt[df]
        print "      full",opt[df]
        print "      sub ",dopt[df]

        ff = h5py.File(opt[df], 'r')
        sf = h5py.File(dopt[df], 'w')

        ndset = len(ff)
        assert ndset == 1

        dsetName = [ dInFile for dInFile in ff][0]
        sl = slicing
        subFile = ff[dsetName].value[sl[0][0]:sl[0][1], sl[1][0]:sl[1][1], sl[2][0]:sl[2][1], ]

        if df == "oversegL0" or df == "oversegL1":
            subFile = vigra.analysis.labelVolume(subFile.astype(numpy.uint32))

        print "    save:"
        sf[dsetName] = subFile

        ff.close()
        sf.close()

        #fullData  = vigra.impex.readHDF5()