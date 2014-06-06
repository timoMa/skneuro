from _utilities import *
from cStringIO import StringIO
import sys

def boostPythonInjector(clsToExtend):
    class Injector(object):
        class __metaclass__(clsToExtend.__class__):
            def __init__(self, name, bases, dict):
                for b in bases:
                    if type(b) not in (self, type):
                        for k,v in dict.items():
                            setattr(b, k, v)
                return type.__init__(self, name, bases, dict)
    return Injector


def extendBlocking3d():
    class MoreBlocking3d(boostPythonInjector(Blocking3d), Blocking3d):
        def foo(self):
            print 'foo!'

        def writeBlock(self, block, blockData, totalData):
            # call c++ to unlock gil if possible
            pass

        def readBlock(self, block, totalData, out=None):
            # call c++ to unlock gil if possible
            return out

        def __str__(self):

            old_stdout = sys.stdout
            sys.stdout = mystdout = StringIO()

            for i in xrange(len(self)):
                print  self.__getitem__(i)

            sys.stdout = old_stdout
            return mystdout.getvalue()
extendBlocking3d()
del extendBlocking3d
