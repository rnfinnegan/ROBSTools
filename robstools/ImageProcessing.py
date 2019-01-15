#!/usr/bin/python

"""
Module name: ImageProcessing
Author:      Robert Finnegan
Date:        December 2018
Description:
---------------------------------
- Lung segmentation
- Bounding box operations
- Cropping operations
---------------------------------

"""

from __future__ import print_function
import os, sys

import SimpleITK as sitk
import numpy as np

def main(arguments):
    return True



def ThresholdAndMeasureLungVolume(image, l=0, u=1):
    # Perform the threshold
    imThresh = sitk.Threshold(image, lower=l, upper=u)
    mask = sitk.ConnectedComponent(sitk.Cast(imThresh*1024, sitk.sitkInt32),True)

    cts = np.bincount(sitk.GetArrayFromImage(mask).flatten())
    maxVals = cts.argsort()[-6:][::-1]

    PBR = np.zeros_like(maxVals, dtype=np.float32)
    NP = np.zeros_like(maxVals, dtype=np.float32)
    label_shape_analysis = sitk.LabelShapeStatisticsImageFilter()
    for i, val in enumerate(maxVals):
        binaryVol = sitk.Equal(mask, val)
        label_shape_analysis.Execute(binaryVol)
        PBR[i] = label_shape_analysis.GetPerimeterOnBorderRatio(True)
        NP[i] = label_shape_analysis.GetNumberOfPixels(True)

    return NP, PBR, mask, maxVals



def AutoLungSegment(image, l = 0.05, u = 0.4, NPthresh=1e5):
    imNorm = sitk.Normalize(sitk.Threshold(image, -1000,500, outsideValue=-1000))
    NP, PBR, mask, labels = ThresholdAndMeasureLungVolume(imNorm,l,u)
    indices = np.array(np.where(np.logical_and(PBR<=5e-4, NP>NPthresh)))

    if indices.size==0:
        print("     Warning - non-zero perimeter/border ratio")
        indices = np.argmin(PBR)

    if indices.size==1:
        validLabels = labels[indices]
        maskBinary = sitk.Equal(mask, int(validLabels))

    else:
        validLabels = labels[indices[0]]
        maskBinary = sitk.Equal(mask, int(validLabels[0]))
        for i in range(len(validLabels)-1):
            maskBinary = sitk.Add(maskBinary, sitk.Equal(mask, int(validLabels[i+1])))
    maskBinary = maskBinary>0
    label_shape_analysis = sitk.LabelShapeStatisticsImageFilter()
    label_shape_analysis.Execute(maskBinary)
    maskBox = label_shape_analysis.GetBoundingBox(True)

    return maskBox, maskBinary



def CropImage(image, maskBox):
    imCrop = sitk.RegionOfInterest(image, size=maskBox[3:], index=maskBox[:3])
    return imCrop



if __name__ == '__main__':
    main()
