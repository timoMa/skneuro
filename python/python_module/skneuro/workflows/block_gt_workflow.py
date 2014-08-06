
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

def pbar(size):
    bar = progressbar.ProgressBar(maxval=size-1, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    return bar


def timeStamp(name):
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    return name+'\n'+st


def writeLock(filepath, name):
    f = open(filepath, 'w')
    f.write(name)

class LazyCalc(object):
    def __init__(self, workFolder):

        # check if path exists
        self.workFolder = workFolder

        if not os.path.exists(workFolder):
            os.makedirs(directory)

        # make dataset creation timestamp

def makeDir(path):
    if path[-1]!='/':
        path+='/'
    if not os.path.exists(path):
            os.makedirs(path)
    return path


class NoWorkToDoExecption(Exception):
    def __init__(self, message):

        # Call the base class constructor with the parameters it needs
        Exception.__init__(self, message)

        # Now for your custom code...
        #self.errors = errors



class LazyCalc:
    def __init__(self, workFolder, verbose=True):
        self.workFolder = workFolder
        self.verbose = verbose


    def lazyCalc(self,name, force=False):

        verb = self.verbose
        class LazyCalc_:    
            def __init__(self, name ,workFolder, force):
                self.name = name 
                self.verbose = verb
                self.jobDoneFile =  workFolder+name+"_is_done.job"
                self.calledTrace = False
                self.force = force
            def __enter__(self):
                self.start = time.clock()

                if self.verbose:
                    print(colored( '%s ...'%self.name, 'cyan'))
                    #print self.name, "..."
                

                if self.force == False and os.path.exists(self.jobDoneFile):
                    print(colored( 'skip computation', 'green'))
                    sys.settrace(lambda *args, **keys: None)
                    frame = inspect.currentframe(1)
                    frame.f_trace = self.trace
                
                else :
                   print(colored( 'do computation', 'red'))


                
                return self
            def trace(self, frame, event, arg):
                self.calledTrace = True
                raise #Exception()
            def __exit__(self, etype, value, traceback):

                if etype is not None and self.calledTrace == False:
                    raise RuntimeError(str(etype)+str(value)+str(traceback))


                self.end = time.clock()
                self.interval = self.end - self.start
                if self.verbose  :
                    print(colored( '... took %f sec\n'%self.interval,  'cyan'))


                # write that job is done
                writeLock(self.jobDoneFile,"%s_is_done"%self.name)


                return True



            def isDone(self):
                return False


        return LazyCalc_(name, self.workFolder, force)







def pngFilesToHdf5(pngFolder, h5File, h5Dset='data',dtype = numpy.uint8 ):

    def tryint(s):
        try:
            return int(s)
        except:
            return s

    def alphanum_key(s):
        """ Turn a string into a list of string and number chunks.
            "z23a" -> ["z", 23, "a"]
        """
        return [ tryint(c) for c in re.split('([0-9]+)', s) ]

    pngFiles = [pngFolder+f for f in os.listdir(pngFolder) if f.endswith('.png')]
    pngFiles.sort(key=alphanum_key)
    
    aFile  =pngFiles[0]
    image = vigra.impex.readImage(aFile).squeeze()
    shape =  image.shape

    nImages = len(pngFiles)
    totalShape  = shape + (nImages,)

    totalData = numpy.ndarray(totalShape, dtype=dtype)
    bar = pbar(nImages)
    for z, filename in enumerate(pngFiles):
        image = vigra.impex.readImage(filename).squeeze()
        totalData[:, :, z] = image[:, :]
        bar.update(z)

    vigra.impex.writeHDF5(totalData, h5File, h5Dset)




def neuro_proof_workflow(inputFolder, workFolder ):

    lazyCalculator = LazyCalc(workFolder)

    sample_names = OrderedDict()
    sample_names["ptrain"]="training_sample1"
    sample_names["train"]="training_sample2"
    sample_names["test"]="validation_sample"


    # compute raw data as hdf5
    for key in sample_names:
        sampleName = sample_names[key]
        sampleFolder = makeDir(workFolder+'%s/'%sampleName)
        with lazyCalculator.lazyCalc("stack_to_hdf5_%s"%sampleName,False):
            resutFile = makeDir(sampleFolder)+"raw.h5"
            pngFilesToHdf5(inputFolder+'%s/grayscale_maps/'%sampleName,resutFile )
    
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


if __name__ == "__main__":
    neuro_proof_workflow(
    "/home/tbeier/src/neuroproof_examples/",
    "/home/tbeier/src/neuroproof_results/"
    )
