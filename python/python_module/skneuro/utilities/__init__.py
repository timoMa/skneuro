from _utilities import *
from thread_pool import ThreadPool
from blockwise_caller import blockwiseCaller 
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
        def __str__(self):

            old_stdout = sys.stdout
            sys.stdout = mystdout = StringIO()

            for i in xrange(len(self)):
                print  self.__getitem__(i)

            sys.stdout = old_stdout
            return mystdout.getvalue()
extendBlocking3d()
del extendBlocking3d





