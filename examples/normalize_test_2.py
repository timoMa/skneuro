import vigra
import numpy
from collections import OrderedDict
def normByQuantile(a,qs):
    a = numpy.clip(a,qs[0],qs[1])
    a -=a.min()
    a /-a.max()
    return a

fFront = "/media/tbeier/309AF4254C0E5431/hhess_2nm/complete_dataset/2x2x2nm/2x2x2nm.0500.tif"
fMid = "/media/tbeier/309AF4254C0E5431/hhess_2nm/complete_dataset/2x2x2nm/2x2x2nm.2500.tif"
fBack = "/media/tbeier/309AF4254C0E5431/hhess_2nm/complete_dataset/2x2x2nm/2x2x2nm.3275.tif"

paths = [fFront,fMid,fBack]
imgs = [vigra.readImage(p) for p in paths]
nimgs = []
for img in imgs:
    s3 = [int(s/10) for s in img.shape[0:2]]
    print s3
    vigra.sampling.resize(img.astype('float32'), s3)

    #imgSmoothed = vigra.filters.gaussianSmoothing(img,sigma=5.0)
    imgSmoothed = img
    lq = vigra.filters.discRankOrderFilter(imgSmoothed.astype('uint8'),50, 0.01)
    hq = vigra.filters.discRankOrderFilter(imgSmoothed.astype('uint8'),50, 0.99)

    #lq = vigra.filters.gaussianSmoothing(lq.astype('float32'),sigma=10.0)
    #hq = vigra.filters.gaussianSmoothing(hq.astype('float32'),sigma=10.0)

    lq = vigra.sampling.resize(lq.astype('float32'), img.shape[0:2])
    hq = vigra.sampling.resize(hq.astype('float32'), img.shape[0:2])


    #imgDict= OrderedDict()
    #imgDict["lq"]=(lq,("img"))
    #imgDict["hq"]=(hq,("img"))
    #imgDict["img"]=(img,("img"))
    #vigra.multiImshow(imgDict,shape=[2,2],show=True)


    imgS = img.copy()

    w = img<lq
    imgS[w] = lq[w]

    w = imgS>hq
    imgS[w] = hq[w]

    imgS-=lq
    imgS/=(hq-lq)

    nimgs.append(imgS)

imgDict= OrderedDict()
imgDict["raw_front"]=(imgs[0],("img"))
imgDict["nraw_front"]=(nimgs[0],("img"))
imgDict["raw_mid"]=(imgs[1],("img"))
imgDict["nraw_mid"]=(nimgs[1],("img"))
imgDict["raw_back"]=(imgs[2],("img"))
imgDict["nraw_back"]=(nimgs[2],("img"))

vigra.multiImshow(imgDict,shape=[2,3],show=True)
