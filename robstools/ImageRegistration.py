#!/usr/bin/python

"""
Module name: ImageRegistration
Author:      Robert Finnegan
Date:        December 2018
Description:
---------------------------------
- Rigid alignment
- Demons DIR
- BSplines DIR
- Label Propagation
---------------------------------

"""


from __future__ import print_function
import os, sys

import SimpleITK as sitk
import numpy as np


def RigidReg(fixedImage, movingImage, parFile, sgFlag=0):

	rigidElastix = sitk.ElastixImageFilter()
	print(fixedImage.GetSize())
	print(movingImage.GetSize())

	if sgFlag==1:
		print("Normalising images")
		fMax = sitk.GetArrayFromImage(fixedImage).max()
		mMax = sitk.GetArrayFromImage(movingImage).max()
		fixedImage = sitk.Cast(fixedImage, sitk.sitkFloat32)/fMax
		movingImage = sitk.Cast(movingImage, sitk.sitkFloat32)/mMax

	rigidElastix.SetFixedImage(fixedImage)
	rigidElastix.SetMovingImage(movingImage)
	rigidElastix.LogToConsoleOn()

	try:
		rigidParameterMap = sitk.ReadParameterFile(parFile)
	except:
		print('No transform parameter found in current directory, searching library...')
		try:
			rigidParameterMap = sitk.ReadParameterFile(parFile)
		except:
			print('No transform parameter in library, using default.')
			parName = MABASDir+'RigidTransformParameters.txt'
			rigidParameterMap = sitk.ReadParameterFile(parFile)

	rigidElastix.SetParameterMap(rigidParameterMap)

	if sgFlag==1:
		rigidElastix.SetParameter(0, 'FinalBSplineInterpolationOrder','1')
		rigidElastix.SetParameter(0, 'DefaultPixelValue','0')

	rigidElastix.SetLogToFile(False)

	rigidElastix.Execute()
	registeredImage = rigidElastix.GetResultImage()
	tfm = rigidElastix.GetTransformParameterMap()

	if sgFlag==1:
		registeredImage = sitk.Cast(registeredImage, sitk.sitkFloat32)
		registeredImage = sitk.Threshold(registeredImage, lower=1e-5, upper=100)

	registeredImage.CopyInformation(fixedImage)

	return sitk.Cast(registeredImage, movingImage.GetPixelID()), tfm[0]



def main(arguments):
    return True

if __name__ == '__main__':
    main()
