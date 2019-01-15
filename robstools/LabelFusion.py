#!/usr/bin/python

"""
Module name: LabelFusion
Author:      Robert Finnegan
Date:        December 2018
Description:
---------------------------------
- UWV, GWV, LWV, BWV, STAPLE
- Label processing
---------------------------------

"""

from __future__ import print_function
import os, sys

import SimpleITK as sitk
import numpy as np
from functools import reduce

def computeWeightMap(targetImage, movingImage, voteType='local', voteParams={'sigma':2.0, 'epsilon':1E-5}):
	"""
	Computes the weight map
	"""
	targetImage = sitk.Cast(targetImage, sitk.sitkFloat32)
	movingImage = sitk.Cast(movingImage, sitk.sitkFloat32)

	squareDifferenceImage = sitk.SquaredDifference(targetImage, movingImage)
	squareDifferenceImage = sitk.Cast(squareDifferenceImage, sitk.sitkFloat32)

	if voteType.lower()=='majority':
		weightMap = targetImage * 0.0 + 1.0

	elif voteType.lower()=='global':
		factor = voteParams['factor']
		sumSquaredDifference  = sitk.GetArrayFromImage(squareDifferenceImage).sum(dtype=np.float)
		globalWeight = factor / sumSquaredDifference

		weightMap = targetImage * 0.0 + globalWeight

	elif voteType.lower()=='local':
		sigma = voteParams['sigma']
		epsilon = voteParams['epsilon']

		rawMap = sitk.DiscreteGaussian(squareDifferenceImage, sigma*sigma)
		weightMap = sitk.Pow(rawMap + epsilon , -1.0)

	elif voteType.lower()=='block':
		gain = voteParams['gain']
		blockSize = voteParams['blockSize']
		if type(blockSize)==int:
			blockSize = (blockSize,)*targetImage.GetDimension()

		rawMap = sitk.Mean(squareDifferenceImage, blockSize)
		weightMap = (rawMap) ** (-1.0*abs(gain))

	else:
		raise ValueError('Weighting scheme not valid.')

	return sitk.Cast(weightMap, sitk.sitkFloat32)


def combineLabels(weightMapDict, labelListDict, threshold=1e-4):
	"""
	Combine labels using weight maps
	"""

	combinedLabelDict = {}

	caseIdList = list(weightMapDict.keys())
	structureNameList = [list(i.keys()) for i in labelListDict.values()]
	structureNameList = np.unique([item for sublist in structureNameList for item in sublist] )

	for structureName in structureNameList:
		# Find the cases which have the strucure (in case some cases do not)
		validCaseIdList = [i for (i,j) in list(labelListDict.items()) if structureName in j.keys()]

		# Get valid weight images
		weightImageList = [weightMapDict[caseId] for caseId in validCaseIdList]

		# Sum the weight images
		weightSumImage = reduce(lambda x,y:x+y, weightImageList)
		weightSumImage = sitk.Mask(weightSumImage, weightSumImage==0, maskingValue=1, outsideValue=1)

		# Combine weight map with each label
		weightedLabels = [weightMapDict[caseId]*sitk.Cast(labelListDict[caseId][structureName], sitk.sitkFloat32) for caseId in validCaseIdList]

		# Combine all the weighted labels
		combinedLabel = reduce(lambda x,y:x+y, weightedLabels) / weightSumImage

		# Normalise
		combinedLabel = sitk.RescaleIntensity(combinedLabel, 0, 1)

		# Threshold - grants vastly improved compression performance
		if threshold:
			combinedLabel = sitk.Threshold(combinedLabel, lower=threshold, upper=1, outsideValue=0.0)

		combinedLabelDict[structureName] = combinedLabel

	return combinedLabelDict

def processProbabilityImage(probabilityImage, threshold=0.5):

	# Get the starting binary image
	binaryImage = sitk.BinaryThreshold(probabilityImage, lowerThreshold=threshold)

	# Fill holes
	binaryImage = sitk.BinaryFillhole(binaryImage)

	# Apply the connected component filter
	labelledImage = sitk.ConnectedComponent(binaryImage)

	# Measure the size of each connected component
	labelShapeFilter = sitk.LabelShapeStatisticsImageFilter()
	labelShapeFilter.Execute(labelledImage)
	labelIndices = labelShapeFilter.GetLabels()
	voxelCounts  = [labelShapeFilter.GetNumberOfPixels(i) for i in labelIndices]
	if voxelCounts==[]:
		return binaryImage

	# Select the largest region
	largestComponentLabel = labelIndices[np.argmax(voxelCounts)]
	largestComponentImage = (labelledImage==largestComponentLabel)

	return largestComponentImage


def main(arguments):
	return True

if __name__ == '__main__':
	main()
