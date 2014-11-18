import skneuro
import vigra
from skneuro import workflows as wf
import numpy
import h5py

from vigra import graphs

optJsonFile = "opt.json"
opt = wf.loadJson(optJsonFile)

grayData = []
segData  = []



x = [0,200]
y = [0,500]
z = [0,500]

xx = (x[1]-x[0])/2

print "read raw data"
f = h5py.File(opt['rawData'], "r")
rawData = f[opt['rawDatasetName']][x[0]:x[1], y[0]:y[1], z[0]:z[1]]
f.close()


if False :


    f = h5py.File(opt['boundaryP1'], "r")
    boundaryP1 = f['exported_data'][x[0]:x[1], y[0]:y[1], z[0]:z[1],:]
    f.close()

    f = h5py.File(opt['rawSemanticP0'], "r")
    rawSemanticP0 = f['data'][x[0]:x[1], y[0]:y[1], z[0]:z[1],:]
    f.close()



    for c in range(5):
        cimg = rawSemanticP0[xx, :, :, c]
        vigra.impex.writeImage(cimg,"for_anna/semantic_%d.png"%c)
        vigra.impex.writeImage(1.0-cimg,"for_anna/isemantic_%d.png"%c)

    img = boundaryP1[xx,:, :, 0]
    vigra.impex.writeImage(img,"for_anna/p.png")
    vigra.impex.writeImage(1.0-img,"for_anna/ip.png")

    print "show"
    grayData.append((rawData, 'raw'))
    grayData.append((boundaryP1, 'boundaryP1'))
    grayData.append((rawSemanticP0, 'rawSemanticP0'))
    skneuro.addHocViewer(grayData, segData)


if True:


    raw = rawData[xx,:,:]
    raw = raw.reshape((raw.shape[0],raw.shape[1],1))
    raw = numpy.concatenate([raw, raw, raw],axis=2)

    rawIn = raw.copy()
    print raw.shape

    for l in  range(2):

        raw = rawIn.copy()
        l=int(l)

        f = h5py.File(opt["oversegL%d"%l], "r")
        segRaw = f['data'][x[0]:x[1], y[0]:y[1], z[0]:z[1]]
        f.close()

        seg = segRaw[xx,:, :]
        mSeg = seg.max()

        colors = numpy.random.rand(3*(mSeg+1)).reshape(mSeg+1,3)*255.0

        fseg  = seg.reshape(-1)

        fcolorseg = colors[fseg,:]
        fcolorseg = fcolorseg.reshape([seg.shape[0], seg.shape[1], 3])

        alpha = 0.5
        segAndRaw  = (1.0 - alpha)*raw + alpha*fcolorseg
        vigra.impex.writeImage(segAndRaw,"for_anna/seg%d.png"%l)



        raw = vigra.taggedView(raw,'xyc')



        niceImg = vigra.segShow(img=raw.astype(numpy.float32).squeeze(),labels=vigra.analysis.labelImage(seg),
            edgeColor=(0.1,0.9,0.2),
            alpha=0.3,
            show=True,
            returnImg=True,
            r=1)
        
        vigra.impex.writeImage(niceImg.astype(numpy.float32),"for_anna/nseg%d.png"%l)