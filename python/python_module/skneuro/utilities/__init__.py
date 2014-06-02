from _utilities import *

def extendBlocking3d():
    clsToExtend = Blocking3d
    class Injector(object):
        class __metaclass__(clsToExtend.__class__):
            def __init__(self, name, bases, dict):
                for b in bases:
                    if type(b) not in (self, type):
                        for k,v in dict.items():
                            setattr(b,k,v)
                return type.__init__(self, name, bases, dict)


    class MoreCls(Injector, clsToExtend):
        def foo(self):
            print 'foo!'

extendBlocking3d()
del extendBlocking3d
