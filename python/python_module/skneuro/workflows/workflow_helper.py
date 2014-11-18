
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


import json
from pprint import pprint
import re

# Regular expression for comments
comment_re = re.compile(
    '(^)?[^\S\n]*/(?:\*(.*?)\*/[^\S\n]*|/[^\n]*)($)?',
    re.DOTALL | re.MULTILINE
)

def loadJson(filename):
    """ Parse a JSON file
        First remove comments and then use the json module package
        Comments look like :
            // ...
        or
            /*
            ...
            */
    """
    with open(filename) as f:
        content = unicode(''.join(f.readlines())).encode('ascii')

        ## Looking for comments
        match = comment_re.search(content)
        while match:
            # single line comment
            content = content[:match.start()].encode('ascii') + content[match.end():].encode('ascii')
            match = comment_re.search(content)


        #print content

        # Return json file
        return json.loads(content)




def pbar(size):
    bar = progressbar.ProgressBar(maxval=size, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
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
    


    aFile  =pngFiles[0].encode('ascii')
    image = vigra.impex.readImage(aFile).squeeze()
    shape =  image.shape

    nImages = len(pngFiles)
    totalShape  = shape + (nImages,)

    totalData = numpy.ndarray(totalShape, dtype=dtype)

    bar = pbar(nImages)
    for z, filename in enumerate(pngFiles):
        image = vigra.impex.readImage(filename.encode('ascii')).squeeze()
        totalData[:, :, z] = image[:, :]
        bar.update(z)
    vigra.impex.writeHDF5(totalData, h5File, h5Dset)




if __name__ == "__main__":
    neuro_proof_workflow(
    "/home/tbeier/src/neuroproof_examples/",
    "/home/tbeier/src/neuroproof_results/"
    )
