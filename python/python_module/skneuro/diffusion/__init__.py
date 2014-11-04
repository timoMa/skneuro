import vigra
import numpy 
from skneuro.blockwise_filters import blockwiseHessianOfGaussianSortedEigenvalues,
                                      blockwiseGaussianGradientMagnitude

from scipy.ndimage.filters import gaussian_filter1d


class DiffusionOpt(object):
    def __init__(self):
        self.maxT  = 10.0
        self.deltaT = 0.1
        self.sigmaTensor = 2.0
        self.sigmaGrad = 1.0

class DiffScheme3DRotationInvariant(object):

    def __init__(self, img, options):
        self.shape = img.shape
        self.options = self.options


    def step(self, img, D): 

        Dxx, Dyy, Dzz, Dxy, Dxz, Dyz = D

        # compute the flux
        gx, gy, gz = self.getGradients(img)
        j0 = Dxx * gx + Dxy * gy + Dxz * uz
        j1 = Dxy * gx + Dyy * gy + Dyz * uz
        j2 = Dxz * gx + Dyz * gy + Dzz * uz

        # make boarder zero
        for j in [j0, j1, j2]:
            j[:,  :,  0] = 0 
            j[:,  :, -1] = 0 
            j[:,  0,  :] = 0 
            j[:, -1,  :] = 0 
            j[ 0, :,  :] = 0 
            j[-1, :,  :] = 0 

        # compute the gradients on flux
        dj0 = gaussian_filter1d(j0, simga=self.options.sigmaGrad, axis=0, order=1)
        dj1 = gaussian_filter1d(j1, simga=self.options.sigmaGrad, axis=1, order=1)
        dj2 = gaussian_filter1d(j2, simga=self.options.sigmaGrad, axis=2, order=1)

        dj = dj0 + dj1 + dj2

        return img + dj*self.options.deltaT

    def getGradients(self, img):
        grads = blockwiseGaussianGradientMagnitude(img, self.options.sigmaGrad)
        gx = grads[:,:,:, 0]
        gy = grads[:,:,:, 1]
        gz = grads[:,:,:, 2]
        return (gx, gy, gz)

class CoherenceFilterStep3D(class):
    def __init__(self, img, options):
        self.shape = img.shape
        self.options = self.options
        self.eigenmode = 'plane'
        self.diffScheme = DiffScheme3DRotationInvariant(img, options)

    def step(self, img):

        # compute the tensor
        tensor = self.getTensor(img)

        # get diffusion tensor
        D  = getDiffusionTensor(tensor)

        return self.diffScheme(img, D)

    def getTensor(self, img):
        tensor = blockwiseHessianOfGaussianSortedEigenvalues(img, options.sigmaTensor)
        return tensor

    def getDiffusionTensor(self, tensor):
        e0, e1, e1,  v0, v1, v2  = self.getEigenvalueEigenvector(tensor)

        if(options.eigenmode == 'plane'):
            pass

        Dxx = e0*v0[0]*v[0] + e1*v1[0]*v1[0] + e2*v2[0]*v2[0] 
        Dyy = e0*v0[1]*v[1] + e1*v1[1]*v1[1] + e2*v2[1]*v2[1] 
        Dzz = e0*v0[2]*v[2] + e1*v1[2]*v1[2] + e2*v2[2]*v2[2] 
        Dxy = e0*v0[0]*v[1] + e1*v1[0]*v1[1] + e2*v2[0]*v2[1] 
        Dxz = e0*v0[0]*v[2] + e1*v1[0]*v1[2] + e2*v2[0]*v2[2] 
        Dyz = e0*v0[1]*v[2] + e1*v1[1]*v1[2] + e2*v2[1]*v2[2] 

        return (Dxx, Dyy, Dzz, Dxy, Dxz, Dyz)

    def getEigenvalueEigenvector(self, tensor):
        pass

def coherence_filter(img, options):

    stepMaker = CoherenceFilterStep3D(img=img, options=options)

    # main diffusion loop
    while(t<options.maxT):  
        
        img = stepMaker.step(img)
