#!/usr/bin/env python
import SimpleITK as sitk
import os
import numpy as np
from Utilities import DataStorage, RegUtils
import gc
import scipy.ndimage as ndimage
import scipy.signal as signal
import sys
from select import select
import matplotlib.pyplot as plt
import scipy.ndimage

def calcWeightMaps(caseTarget, caseMovingList, voteType='local', footprintType='uniform', footprintSize=7):
	"""
	Computes the weight maps
	"""
	if voteType.lower()=='global':
		weightMaps={}
		targetNDA = sitk.GetArrayFromImage(caseTarget.image)
		for movingCase in caseMovingList:
			movingNDA = sitk.GetArrayFromImage(movingCase.image)
			SD = (targetNDA-movingNDA)**2
			SSD = 1e12 * np.sum(SD)**(-1.)
			weightMaps[movingCase.id] = SSD
			print('Weight [{0} -> {1}] = {2:3f}'.format(movingCase.id,caseTarget.id, SSD))

	if voteType.lower()=='local':
		weightMaps={}
		#isoTarget =RegUtils.isotropicResample(caseTarget.image,2)
		#targetNDA = sitk.GetArrayFromImage(isoTarget)
		targetNDA = sitk.GetArrayFromImage(caseTarget.image)

		kernelDict = {'linear':LinearFootprint, 'uniform':UniformFootprint, 'gaussian':GaussianFootprint}
		kernel = kernelDict[footprintType.lower()]
		footprint = kernel(size=footprintSize, sigma=2.0)

		print 'Using footprint: {0}'.format(footprintType.capitalize())

		for movingCase in caseMovingList:

			#isoMoving = RegUtils.isotropicResample(movingCase.image,2)
			#movingNDA = sitk.GetArrayFromImage(isoMoving)
			movingNDA = sitk.GetArrayFromImage(movingCase.image)
			#print movingCase.id
			#print movingCase.image.GetSize()
			#print caseTarget.image.GetSize()

			SD = (targetNDA-movingNDA)**2.0
			neighbourSum = (signal.fftconvolve(SD, footprint, mode='same')+1e-5)**(-1.0)
			print('Weight [{0} -> {1}] = {2:.2f}'.format(movingCase.id,caseTarget.id, -np.log(np.mean(neighbourSum))))

			weightMaps[movingCase.id]=neighbourSum


	if voteType.lower()=='patch':
		weightMaps={}
		#isoTarget =RegUtils.isotropicResample(caseTarget.image,2)
		#targetNDA = sitk.GetArrayFromImage(isoTarget)
		targetNDA = sitk.GetArrayFromImage(caseTarget.image)

		footprint = UniformFootprint(size=footprintSize)

		print 'Using footprint: {0}'.format(footprintType.capitalize())

		for movingCase in caseMovingList:

			#isoMoving = RegUtils.isotropicResample(movingCase.image,2)
			#movingNDA = sitk.GetArrayFromImage(isoMoving)
			movingNDA = sitk.GetArrayFromImage(movingCase.image)

			SD = (targetNDA-movingNDA)**2
			neighbourSum = (signal.fftconvolve(SD, footprint, mode='same'))/(footprintSize**3)
			print np.std(neighbourSum), np.mean(neighbourSum), neighbourSum.max(), neighbourSum.min()

			print('Weight [{0} -> {1}] = {2:3f}'.format(movingCase.id,caseTarget.id, np.sum(neighbourSum)))
			weightMaps[movingCase.id]=neighbourSum

	return weightMaps




def WeightedVote(caseTarget, caseMovingList, structList, voteType,  weightMaps=None, saveBinary=True, saveProbImage=True, saveSmoothBinary=True, baseFileName=None, footprintSize=7):
	"""
	Uses a selection of:
		- local weighted voting
		- global weighted voting
		(hopefully more soon)
	to combine labels.
	"""

	if voteType.lower()=='local' or voteType.lower()=='patch':
		# Ensure the sum of weights in each pixel is 1
		print('Locally weighted voting regime.')

		if voteType.lower()=='patch':
			print('Patch-based weight estimation.')
			# Local adaptation of decay parameter
			h = np.mean(weightMaps.values(), axis=0)
			print('Calibration of decay parameter:')
			print h.shape, h.mean(), h.std()
			print('Calculating locally adapted weight maps.')
			for m in weightMaps:
				#Exponential
				#weightMaps[m] = np.exp(-1.0*weightMaps[m]/h)

				#Linear
				#weightMaps[m] = 1-(weightMaps[m]-h)/dh

				#Power
				#weightMaps[m] = (1-(weightMaps[m]-h)/dh)**2
				weightMaps[m] = (weightMaps[m])**-6.0


		print('Voxel-wise normalisation')
		weightSumArray = np.sum(weightMaps.values(), axis=0)
		weightSumArray[weightSumArray==0]=1.0
		for m in weightMaps:
			normWeightArray = weightMaps[m]
			normWeightArray[~np.isfinite(normWeightArray)]=0
			normWeightArray /= weightSumArray
			weightMaps[m] = normWeightArray


	elif voteType.lower()=='global':
		print('Globally weighted voting regime.')
	else:
		raise ValueError('Weighting scheme not valid.')

	finalProbMaps = {}
	for struct in structList:
		print('Combining images for structure: {0}'.format(struct))

		# Find the cases which have the strucure (in case some cases do not)
		accIndices = [i for i in range(len(caseMovingList)) if struct in caseMovingList[i].structures.keys()]
		print accIndices
		accCases = [caseMovingList[i] for i in accIndices]

		# Combine weight map with each label
		weightedLabels = [sitk.GetArrayFromImage(m.structures[struct])*weightMaps[m.id] for m in accCases]

		# Combine all the weighted labels
		combinedLabel = np.sum(weightedLabels, axis=0)

		# Normalise
		combinedLabel /= np.max(combinedLabel)


		nonSmoothLabelIm = sitk.GetImageFromArray(combinedLabel)
		nonSmoothLabelIm.CopyInformation(caseTarget.image)
		#if voteType.lower()=='local' or voteType.lower()=='patch':
		#	combinedLabel = RegUtils.identicalResample(combinedLabel, caseTarget.image, 2)

		smoothfinalArr = signal.fftconvolve(combinedLabel, GaussianFootprint(size=footprintSize, sigma=0.5), mode='same')
		smoothfinalArr[smoothfinalArr<1e-4] = 0 #vastly improves compression
		smoothfinalIm  = sitk.GetImageFromArray(smoothfinalArr)
		smoothfinalIm.CopyInformation(caseTarget.image)

		mbaseFileName = baseFileName.format(struct)

		th=0.5

		if saveProbImage:
			outputFileName = '{0}_probability.nii.gz'.format(mbaseFileName)
			print("Writing probability image to {0}".format(outputFileName))
			sitk.WriteImage(smoothfinalIm, outputFileName)

		if saveBinary:
			finalLabel = sitk.BinaryThreshold(nonSmoothLabelIm, lowerThreshold=th)
			outputFileName = '{0}.nii.gz'.format(mbaseFileName)
			print("Writing binary image to {0}".format(outputFileName))
			sitk.WriteImage(finalLabel, outputFileName)

		if saveSmoothBinary:
			smoothfinalLabel = sitk.BinaryThreshold(smoothfinalIm, lowerThreshold=th)
			ccif = sitk.ConnectedComponentImageFilter()
			ccif.FullyConnectedOn()
			smoothConnected = ccif.Execute(smoothfinalLabel)
			ind = np.argmax([sitk.GetArrayFromImage(smoothConnected==i).sum() for i in range(1,20)])
			largestComp = (smoothConnected==(ind+1))
			smoothFilled = sitk.BinaryFillhole(largestComp, True)

			outputFileName = '{0}_processed.nii.gz'.format(mbaseFileName)
			print("Writing smoothed binary image to {0}".format(outputFileName))
			sitk.WriteImage(smoothFilled, outputFileName)

		finalProbMaps[struct] = smoothfinalIm
	return finalProbMaps

def MajorityVote(caseTarget, caseMovingList, struct, saveBinary=True, saveProbImage=True, saveSmoothBinary=True, baseFileName=None, footprintSize=7):

	probabilityLabels={}
	# Find the cases which have the strucure (in case some cases do not)
	accIndices = [i for i in range(len(caseMovingList)) if struct in caseMovingList[i].structures.keys()]
	accCases = [caseMovingList[i] for i in accIndices]

	# Combine weight map with each label
	labels = [sitk.GetArrayFromImage(m.structures[struct]) for m in accCases]

	# Combine all the weighted labels
	combinedLabel = 1.0*np.sum(labels, axis=0)

	# Normalise
	combinedLabel /= np.max(combinedLabel)
	nonSmoothLabelIm = sitk.GetImageFromArray(combinedLabel)
	nonSmoothLabelIm.CopyInformation(caseTarget.image)
	#if voteType.lower()=='local' or voteType.lower()=='patch':
	#	combinedLabel = RegUtils.identicalResample(combinedLabel, caseTarget.image, 2)

	smoothfinalArr = signal.fftconvolve(combinedLabel, GaussianFootprint(size=footprintSize, sigma=0.5), mode='same')
	smoothfinalArr[smoothfinalArr<1e-4] = 0 #vastly improves compression
	smoothfinalIm  = sitk.GetImageFromArray(smoothfinalArr)
	smoothfinalIm.CopyInformation(caseTarget.image)

	mbaseFileName = baseFileName.format(struct)

	th=0.5

	if saveProbImage:
		outputFileName = '{0}_probability.nii.gz'.format(mbaseFileName)
		print("Writing probability image to {0}".format(outputFileName))
		sitk.WriteImage(smoothfinalIm, outputFileName)

	if saveBinary:
		finalLabel = sitk.BinaryThreshold(nonSmoothLabelIm, lowerThreshold=th)
		outputFileName = '{0}.nii.gz'.format(mbaseFileName)
		print("Writing binary image to {0}".format(outputFileName))
		sitk.WriteImage(finalLabel, outputFileName)

	if saveSmoothBinary:
		smoothfinalLabel = sitk.BinaryThreshold(smoothfinalIm, lowerThreshold=th)
		ccif = sitk.ConnectedComponentImageFilter()
		ccif.FullyConnectedOn()
		smoothConnected = ccif.Execute(smoothfinalLabel)
		ind = np.argmax([sitk.GetArrayFromImage(smoothConnected==i).sum() for i in range(1,10)])
		largestComp = (smoothConnected==(ind+1))
		smoothFilled = sitk.BinaryFillhole(largestComp, True)

		outputFileName = '{0}_processed.nii.gz'.format(mbaseFileName)
		print("Writing smoothed binary image to {0}".format(outputFileName))
		sitk.WriteImage(smoothFilled, outputFileName)
	return 1

def STAPLECombine(caseTarget, caseMovingList, struct, threshold=0.5, saveBinary=True, saveProbImage=True, saveSmoothBinary=True, baseFileName=None, footprintSize=7):

	probabilityLabels={}
	# Find the cases which have the strucure (in case some cases do not)
	accIndices = [i for i in range(len(caseMovingList)) if struct in caseMovingList[i].structures.keys()]
	accCases = [caseMovingList[i] for i in accIndices]

	# Assume all labels have the same range of values (e.g. 0->1)
	f = sitk.MinimumMaximumImageFilter()
	f.Execute(accCases[0].structures[struct])
	max1 = f.GetMaximum()
	f.Execute(accCases[1].structures[struct])
	max2 = f.GetMaximum()

	"""
	if max1!=max2:
		print("Label intensities have different ranges, check input.")
		print("Label image A: {0}\nLabel image B: {1}".format(max1,max2))
		print("Type a,b,n (where n is another number) to select appropriate maximum label intensity.")
		rlist,_,_ = select([sys.stdin],[],[],3)
		if rlist:
			sel = sys.stdin.readline()
			if sel.strip().lower()=='b':
				max1=max2
			elif sel.strip().lower()!='a' and sel.strip().lower()!='b':
				try:
					sel = float(sel.strip())
				except:
					print("Choice must be a number.")
					exit()
		else:
			print("Timeout - selecting highest value.")
			max1 = max([max1,max2])
	"""
	max1 = max([max1,max2])

	threshVal = threshold*max1
	print("Given threshold of {0} - image intensity threshold = {1}.".format(threshold, threshVal))

	# STAPLE requires binary labels - hence the need to compute the binary threshold
	binLabels = [m.structures[struct]>threshVal for m in accCases]

	# Combine all the weighted labels
	nonSmoothLabelIm = sitk.STAPLE(binLabels)
	nonSmoothLabelIm.CopyInformation(caseTarget.image)

	smoothfinalArr = signal.fftconvolve(sitk.GetArrayFromImage(nonSmoothLabelIm), GaussianFootprint(size=footprintSize, sigma=0.5), mode='same')
	smoothfinalArr[smoothfinalArr<1e-4] = 0 #vastly improves compression
	smoothfinalIm  = sitk.GetImageFromArray(smoothfinalArr)
	smoothfinalIm.CopyInformation(caseTarget.image)

	mbaseFileName = baseFileName.format(struct)

	th=0.5

	if saveProbImage:
		outputFileName = '{0}_probability.nii.gz'.format(mbaseFileName)
		print("Writing probability image to {0}".format(outputFileName))
		sitk.WriteImage(smoothfinalIm, outputFileName)

	if saveBinary:
		finalLabel = sitk.BinaryThreshold(nonSmoothLabelIm, lowerThreshold=th)
		outputFileName = '{0}.nii.gz'.format(mbaseFileName)
		print("Writing binary image to {0}".format(outputFileName))
		sitk.WriteImage(finalLabel, outputFileName)

	if saveSmoothBinary:
		smoothfinalLabel = sitk.BinaryThreshold(smoothfinalIm, lowerThreshold=th)
		ccif = sitk.ConnectedComponentImageFilter()
		ccif.FullyConnectedOn()
		smoothConnected = ccif.Execute(smoothfinalLabel)
		ind = np.argmax([sitk.GetArrayFromImage(smoothConnected==i).sum() for i in range(1,10)])
		largestComp = (smoothConnected==(ind+1))
		smoothFilled = sitk.BinaryFillhole(largestComp, True)

		outputFileName = '{0}_processed.nii.gz'.format(mbaseFileName)
		print("Writing smoothed binary image to {0}".format(outputFileName))
		sitk.WriteImage(smoothFilled, outputFileName)
	return 1




def UniformFootprint(size=7):
	"""
	Return a uniform footprint
	"""
	return np.ones([size,size,size])

def LinearFootprint(size=7):
	"""
	Return a footprint which drops of linearly as a function of the Euclidean distance from the centre
	"""
	x,y,z=np.mgrid[0:size:1,0:size:1,0:size:1]
	EuclideanDistance = np.sqrt((x-(size-1)/2.)**2 + (y-(size-1)/2.)**2 + (z-(size-1)/2.)**2)
	f =  np.sqrt(3)*(size-1)/2. - EuclideanDistance
	"""
	fig=plt.figure()
	for i in range(9):
		ax=fig.add_subplot(3,3,i+1)
		ax.imshow(f[i], interpolation='none')
	fig.show()
	"""
	return f

def GaussianFootprint(size=7, sigma='auto'):
	"""
	Return a Gaussian footprint, centered in the 3D centre of the region
	The scale (standard deviation) can either be given or calculated automatically
	"""
	x,y,z=np.mgrid[0:size:1,0:size:1,0:size:1]
	d = (x-(size-1)/2.)**2 + (y-(size-1)/2.)**2 + (0.5*(z-(size-1)/2.))**2

	if str(sigma).lower()=='auto':
		sigma = size/2.0

	f =  np.exp(-0.5*d/(sigma**2))
	f /= np.sum(f)
	return f
