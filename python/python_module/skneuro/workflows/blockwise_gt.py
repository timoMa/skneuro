from workflow_helper import *
import os
import datetime
import time
import sys
import inspect
import re
import vigra
import progressbar
import numpy

from colorama import init
from colorama import Fore, Back, Style
from termcolor import colored
from collections import OrderedDict
from skneuro import blockwise_filters as blockfilt
init()


def neuroproofWorkflow(opts):



    inputRootFolder = opts['inputRootFolder']
    workRootFolder  = opts['workRootFolder']
    datasetSamples = opts['datasetSamples']

    orderdedSampleNames  = opts['sampleOrder']


    lazyCalculator = LazyCalc(workRootFolder)

    # compute raw data as hdf5
    for sampleName in orderdedSampleNames:

        sampleOpts = datasetSamples[sampleName]
        rawDataFolder = sampleOpts['rawDataFolder']
        rawDataFolder = inputRootFolder + rawDataFolder
       
        sampleWorkFolder = makeDir(workRootFolder + sampleName)

        with lazyCalculator.lazyCalc("stack_to_hdf5_%s"%sampleName,False):
            resutFile = makeDir(sampleWorkFolder)+"raw.h5"
            pngFilesToHdf5(rawDataFolder, resutFile )

    """
    # compute eigenvalues of hessian on raw data
    for key in sample_names:
        sampleName = sample_names[key]
        hessianScales = [4.0]#[1.0, 2.0, 3.0, 4.0]
        



        rawData = vigra.impex.readHDF5(sampleFolder+"raw.h5",'data')
        for scale in hessianScales:
            scaleStr = "{0:.2f}".format(scale)
            with lazyCalculator.lazyCalc("hessian_lew%s_scale_%s"%(sampleName, scale),False):
                result = blockfilt.blockwiseHessianOfGaussianLargestEigenvalues(rawData,scale)
                resultFolder = makeDir(sampleFolder+"hessian_lew")
                resultFile = resultFolder +"scale_%s.h5"%scaleStr
                vigra.impex.writeHDF5(result, resultFile, 'data')
    """
