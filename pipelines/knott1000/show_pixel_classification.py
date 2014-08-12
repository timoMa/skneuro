import skneuro
import vigra
from skneuro import workflows as wf
import numpy

optJsonFile = "opt.json"
opt = wf.loadJson(optJsonFile)

grayData = []
segData  = []

print "read raw data"
rawData = vigra.impex.readHDF5(opt['rawData'], opt['rawDatasetName']).view(numpy.ndarray)
print "read boundary p1"
boundaryP1 = vigra.impex.readHDF5(opt['boundaryP1'],'exported_data').view(numpy.ndarray)
print "read semantic p0"
rawSemanticP0 = vigra.impex.readHDF5(opt['rawSemanticP0'],'data').view(numpy.ndarray)

print "show"
grayData.append((rawData, 'raw'))
grayData.append((boundaryP1, 'boundaryP1'))
grayData.append((rawSemanticP0, 'rawSemanticP0'))
skneuro.addHocViewer(grayData, segData)
