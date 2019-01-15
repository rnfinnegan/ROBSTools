import numpy as np
import SimpleITK as sitk

def curl(vectorFieldArray, delta=[1,1,1], returnComponents=False):
    """
    Calculates the 3D curl
    """
    gradientArray = np.gradient(vectorFieldArray, axis=2)
    deltaFx = gradientArray[:,:,:,0]
    deltaFy = gradientArray[:,:,:,1]
    deltaFz = gradientArray[:,:,:,2]
    deltax,deltay,deltaz = delta
    curlx = (deltaFz/deltay-deltaFy/deltaz)
    curly = (deltaFx/deltaz-deltaFz/deltax)
    curlz = (deltaFy/deltax-deltaFx/deltay)
    if not returnComponents:
        return np.stack((curlx,curly, curlz), axis=3)
    else:
        return curlx, curly, curlz

def divergence(vectorFieldArray, delta=[1,1,1]):
    """
    Calculates the 3D divergence
    """
    gradientArray = np.gradient(vectorFieldArray, axis=2)
    deltaFx = gradientArray[:,:,:,0]
    deltaFy = gradientArray[:,:,:,1]
    deltaFz = gradientArray[:,:,:,2]
    deltax,deltay,deltaz = delta
    deltax,deltay,deltaz = delta
    divx = (deltaFx/deltax)
    divy = (deltaFy/deltay)
    divz = (deltaFz/deltaz)
    return np.sum((divx,divy,divz), axis=0)

def jacobian(vectorFieldImage):
    """
    Calculate the Jacobian image
    """
    try:
        jacImage = sitk.DisplacementFieldJacobianDeterminant(vectorFieldImage)
        return jacImage
    except:
        print "Input is a SimpleITK image"
        return None
