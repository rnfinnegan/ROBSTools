import os
import SimpleITK as sitk
import numpy as np
from Utilities import DataStorage
import gc

codeDir = "/media/HDD/Documents/University/PhD/Research/Software/MedicalImagingTools/Python/Registration"

def command_iteration(filter) :
	print "{0:3} = {1:10.5f}".format(filter.GetElapsedIterations(),filter.GetMetric())

def RigidReg(caseTarget, caseMoving, save=True):

	#rigidElastix = sitk.SimpleElastix()
	rigidElastix = sitk.ElastixImageFilter()
	rigidElastix.SetFixedImage(caseTarget.image)
	rigidElastix.SetMovingImage(caseMoving.image)
	rigidElastix.LogToConsoleOn()

	rigidParameterMap = sitk.ReadParameterFile("/media/HDD/Documents/University/PhD/Research/Software/MedicalImagingTools/Python/Registration/ParameterFiles/RigidTransformParameters.txt")
	rigidElastix.SetParameterMap(rigidParameterMap)

	rigidElastix.Execute()
	registeredImage = rigidElastix.GetResultImage()


	# Save transform to use later
	tfm = rigidElastix.GetTransformParameterMap()

	if save:
		output =  "Rigid/LeaveOut{0}/Case_{1}_to_Case_{0}_rigid.nii.gz".format(caseTarget.id, caseMoving.id)
		tfmFile = "Rigid/LeaveOut{0}/Case_{1}_to_Case_{0}_tfm.txt".format(caseTarget.id, caseMoving.id)

		sitk.WriteImage(registeredImage, output)

		if len(tfm)>1:
			raise ValueError("Only use one transform (or modify the code)")
		else:
			print 'Saving transform file to disk.'
			sitk.WriteParameterFile(tfm[0], tfmFile)

	registeredCase = DataStorage.Case(id=caseMoving.id, image=registeredImage, regType='Rigid')

	return rigidElastix, registeredCase


def DemonsNonRigidReg(caseTarget, caseMoving, Niterations=100, save=False, printFlag=True):

	targetImage = sitk.Cast(caseTarget.image, sitk.sitkFloat32)
	movingImage = sitk.Cast(caseMoving.image, sitk.sitkFloat32)

	demons = sitk.DemonsRegistrationFilter()

	#Here are tunable parameters:
	demons.SetNumberOfIterations( Niterations )
	demons.SetStandardDeviations( 1.0 )

	demons.SmoothDisplacementFieldOn()
	demons.UseMovingImageGradientOn()

	"""
	registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[8,4,0])
    """

	#This code allows monitoring

	if printFlag:
		demons.AddCommand( sitk.sitkIterationEvent, lambda: command_iteration(demons) )

	displacementField = demons.Execute( targetImage, movingImage )

	#Output displacement vector field
	outputTransform = sitk.DisplacementFieldTransform( displacementField )
	deformationField = outputTransform.GetDisplacementField()

	resampler = sitk.ResampleImageFilter()
	resampler.SetReferenceImage(targetImage);
	resampler.SetInterpolator(sitk.sitkLinear)
	resampler.SetDefaultPixelValue(-1024)
	resampler.SetTransform(outputTransform)
	registeredImage = resampler.Execute(movingImage)

	if save:
		output = "Demons/LeaveOut{0}/Case_{1}_to_Case_{0}_nrr.nii.gz".format(caseTarget.id, caseMoving.id)
		outputField = "Demons/LeaveOut{0}/Case_{1}_to_Case_{0}_field.nii.gz".format(caseTarget.id, caseMoving.id)
		sitk.WriteImage(deformationField, outputField)
		sitk.WriteImage(registeredImage, output)

	registeredCase = DataStorage.Case(id=caseMoving.id, image=registeredImage, regType='Demons')

	return outputTransform, registeredCase

def BSplinesNonRigidReg(caseTarget, caseMoving, save=False, parFile='44'):

	targetImage = sitk.Cast(caseTarget.image, sitk.sitkFloat32)
	movingImage = sitk.Cast(caseMoving.image, sitk.sitkFloat32)

	SimpleElastix = sitk.SimpleElastix()
	SimpleElastix.LogToConsoleOn()
	SimpleElastix.SetFixedImage(targetImage)
	SimpleElastix.SetMovingImage(movingImage)
	parMap = sitk.ReadParameterFile(codeDir+'/ParameterFiles/NRR_Par{0}.txt'.format(parFile))
	SimpleElastix.SetParameterMap(parMap)
	registeredImage = SimpleElastix.Execute()

	if save:
		output = "Bsplines/LeaveOut{0}/Case_{1}_to_Case_{0}_nrr.nii.gz".format(caseTarget.id, caseMoving.id)
		outputTfm = "Bsplines/LeaveOut{0}/Case_{1}_to_Case_{0}_bsplines_transform.txt".format(caseTarget.id, caseMoving.id)

		sitk.WriteImage(sitk.Cast(registeredImage, sitk.sitkInt32), output)
		sitk.WriteParameterFile(SimpleElastix.GetTransformParameterMap()[0],outputTfm)

	registeredCase = DataStorage.Case(id=caseMoving.id, image=registeredImage, regType='BSplines')

	return SimpleElastix, registeredCase


def PropagateRegistrationToLabelUsingFile(caseTarget, caseMoving, regType):

	targetIndex = caseTarget.id
	movingIndex = caseMoving.id

	if regType.lower()=='rigid':
		tfmFile = "Rigid/LeaveOut{0}/Case_{1}_to_Case_{0}_tfm.txt".format(targetIndex, movingIndex)

		rigidTransformix = sitk.SimpleTransformix()
		rigidTransformParameterMap = sitk.ReadParameterFile(tfmFile)
		rigidTransformParameterMap["FinalBSplineInterpolationOrder"]= ["0"]
		rigidTransformParameterMap["DefaultPixelValue"] =["0"]
		rigidTransformix.SetTransformParameterMap(rigidTransformParameterMap)

		for s in caseMoving.structures.keys():
		    outputRigid = "Rigid/LeaveOut{0}/Structures/Case_{1}_to_Case_{0}_{2}_rigid.nii.gz".format(targetIndex, movingIndex,s)

		    rigidTransformix.SetMovingImage(caseMoving.structures[s])
		    rigidTransformix.Execute()

		    outputImage = rigidTransformix.GetResultImage()
		    outputImage.CopyInformation(caseTarget.image)
		    sitk.WriteImage(sitk.Cast(outputImage,sitk.sitkUInt8), outputRigid)

		return True

	if regType.lower()=='demons':
		# Now apply the deformation field
		resampler = sitk.ResampleImageFilter()
		resampler.SetReferenceImage(outputImage)
		resampler.SetInterpolator(sitk.sitkNearestNeighbor)
		resampler.SetDefaultPixelValue(0)
		resampler.SetTransform(displacementField)
		out = resampler.Execute(outputImage)


	elif regType.lower()=='bsplines':
		stx = sitk.SimpleTransformix()
		parMapFile = "{2}/LeaveOut{0}/Case_{1}_to_Case_{0}_bsplines_transform.txt".format(caseTarget.id, caseMoving.id, regType.capitalize())

		parMap=sitk.ReadParameterFile(parMapFile)
		parMap['DefaultPixelValue'] = ['0']
		parMap['ResampleInterpolator']=['FinalBSplineInterpolator']
		parMap['FinalBSplineInterpolationOrder']=['0']
		stx.AddTransformParameterMap(parMap)

		for s in caseMoving.structures.keys():
		    outputNonRigid = "Bsplines/LeaveOut{0}/Structures/Case_{1}_to_Case_{0}_{2}_nrr.nii.gz".format(targetIndex, movingIndex,s)
			#Turn this on to save the (vector) deformation field
			#stx.ComputeDeformationFieldOn()
		    stx.SetMovingImage(caseMoving.structures[s])
		    stx.Execute()

		    outputImage = stx.GetResultImage()
		    outputImage.CopyInformation(caseTarget.image)
		    sitk.WriteImage(sitk.Cast(outputImage,sitk.sitkUInt8), outputNonRigid)

		return True

def PropagateRegistrationToLabelUsingTransformation(caseTarget, caseMoving, regType, trans):

	targetIndex = caseTarget.id
	movingIndex = caseMoving.id

	if regType.lower()=='rigid':
		rigidTransformix = trans
		rigidTransform.SetParameter("FinalBSplineInterpolationOrder","0")
		rigidTransform.SetParameter("DefaultPixelValue","0")

		for s in caseMoving.structures.keys():
		    outputRigid = "Rigid/LeaveOut{0}/Structures/Case_{1}_to_Case_{0}_{2}_rigid.nii.gz".format(targetIndex, movingIndex,s)

		    rigidTransformix.SetMovingImage(caseMoving.structures[s])
		    rigidTransformix.Execute()

		    outputImage = rigidTransformix.GetResultImage()
		    outputImage.CopyInformation(caseTarget.image)
		    sitk.WriteImage(sitk.Cast(outputImage,sitk.sitkUInt8), outputRigid)

		return True

	if regType.lower()=='demons':
		# Now apply the deformation field
		resampler = sitk.ResampleImageFilter()
		resampler.SetReferenceImage(outputImage)
		resampler.SetInterpolator(sitk.sitkNearestNeighbor)
		resampler.SetDefaultPixelValue(0)
		resampler.SetTransform(displacementField)
		out = resampler.Execute(outputImage)


	elif regType.lower()=='bsplines':
		stx = trans
		stx.SetTransformParameter("FinalBSplineInterpolationOrder","0")
		stx.SetTransformParameter("DefaultPixelValue","0")

		for s in caseMoving.structures.keys():
		    outputNonRigid = "Bsplines/LeaveOut{0}/Structures/Case_{1}_to_Case_{0}_{2}_nrr.nii.gz".format(targetIndex, movingIndex,s)
			#Turn this on to save the (vector) deformation field
			#stx.ComputeDeformationFieldOn()
		    stx.SetMovingImage(caseMoving.structures[s])
		    stx.Execute()

		    outputImage = stx.GetResultImage()
		    outputImage.CopyInformation(caseTarget.image)
		    sitk.WriteImage(sitk.Cast(outputImage,sitk.sitkUInt8), outputNonRigid)

		return True





def checkReg(indexTarget, indexMoving, dirLoc):
	"""
	Checks that the moving case has been registered successfully.
	"""
	checks = ['Case_{1}_to_Case_{0}_{2}.nii.gz'.format(indexTarget, indexMoving, i) for i in['nrr', 'rigid', 'field']]
	checks.append('Case_{1}_to_Case_{0}_tfm.txt'.format(indexTarget, indexMoving))

	checkFlags = [i in os.listdir(dirLoc) for i in checks]
	if sum(checkFlags)==len(checkFlags):
		return True
	else:
		return False

def checkProp(indexTarget, indexMoving, structList, dirLoc):
	"""
	Checks that all structures in the moving case have been propagated.
	"""
	checks = ['Case_{1}_to_Case_{0}_{2}_{3}.nii.gz'.format(indexTarget, indexMoving, i, j) for j in['nrr', 'rigid'] for i in structList]
	checkFlags = [i in os.listdir(dirLoc) for i in checks]
	if sum(checkFlags)==len(checkFlags):
		return True
	else:
		return False

def main():
	checkExists=True

	fileLoc = "/media/HDD/Documents/University/PhD/Research/CardiacContouring/Data/Cropped/Images"
	structLoc = "/media/HDD/Documents/University/PhD/Research/CardiacContouring/Data/Cropped/Structures"

	structList = ['heart', 'RIGHTATRIUM', 'PULMONICVALVE', 'LEFTVENTRICLE', 'RIGHTATRIUM_2', 'RCORONARYARTERY_2', 'LANTDESCARTERY_2', 'TRICUSPIDVALVE', 'AORTICVALVE_2', 'WHOLEHEART', 'DESCENDINGAORTA', 'PULMONICVALVE_2', 'MITRALVALVE', 'LANTDESCARTERY', 'MITRALVALVE_2', 'PE', 'ContraBreast', 'IpsiLung', 'TRICUSPIDVALVE_2', 'LEFTATRIUM', 'SVC', 'LCIRCUMFLEXARTERY_2', 'LCIRCUMFLEXARTERY', 'LCORONARYARTERY_2', 'RCORONARYARTERY', 'AORTICVALVE_2', 'PULMONARYARTERY', 'WHOLEHEART_2', 'ME', 'PULMONARYARTERY_2', 'ASCENDINGAORTA', 'LCORONARYARTERY', 'SVC_2', 'LEFTVENTRICLE_2', 'CombinedLung', 'AORTICVALVE', 'RIGHTVENTRICLE_2', 'LEFTATRIUM_2', 'DESCENDINGAORTA_2', 'RIGHTVENTRICLE', 'HeartPRV', 'ContraLung', 'ASCENDINGAORTA_2']

	caseListT = ['03','04','05','08','09','10','11','12','13','14','16','17','20','22','23','24','27','32','33','35']
	caseListM = ['03','04','05','08','09','10','11','12','13','14','16','17','20','22','23','24','27','32','33','35']

	for targetIndex in caseListT:
		try:
			os.mkdir('Demons')
		except:
			print "Parent directory exists."

		try:
			os.mkdir("Demons/LeaveOut{0}".format(targetIndex))
			os.mkdir("Demons/LeaveOut{0}/Structures".format(targetIndex))
		except:
			print "Directory for case {0} exists, moving onto next case".format(targetIndex)
			continue

		"""
		Read in the target case
		"""
		targetCase = DataStorage.Case(targetIndex, imageLoc='{0}/Case_{1}_crop.nii.gz'.format(fileLoc, targetIndex))
		for s in structList:
			try:
				targetCase.addStructure(s, imageLoc='{0}/Case_{1}_{2}_crop.nii.gz'.format(structLoc, targetIndex, s))
			except:
				print 'Case {0}, structure {1} not found - continuing'.format(targetCase.id, s)


		for movingIndex in caseListM:
			if checkExists and checkReg(targetIndex, movingIndex, 'Demons/LeaveOut{0}'.format(targetIndex)) and checkProp(targetIndex, movingIndex, structList, 'Demons/LeaveOut{0}/Structures'.format(targetIndex)):
				print 'Image {0} already registered, skipping\n  (Set flag "checkExists" to False to overwrite)'.format(movingIndex)
				continue


			print "Registering case {0} to case {1}".format(movingIndex, targetIndex)

			"""
			Read in the moving case
			"""
			movingCase = DataStorage.Case(id=movingIndex, imageLoc='{0}/Case_{1}_crop.nii.gz'.format(fileLoc, movingIndex))

			for s in structList:
				try:
					movingCase.addStructure(s, imageLoc='{0}/Case_{1}_{2}_crop.nii.gz'.format(structLoc, movingIndex, s))
				except:
					print 'Case {0}, structure {1} not found - continuing'.format(movingCase.id, s)
					continue


			"""
			Perform registrations
			"""
			print "Performing rigid registration."
			rigidMovingCase = RigidReg(targetCase, movingCase)

			print "Performing non rigid (demons) registration."
			displacementField, deformMovingCase = DemonsNonRigidReg(targetCase, rigidMovingCase)


			"""
			Propagate labels through transformations
			"""
			print "Propagating labels."
			for struct in structList:
				try:
					PropagateRegistration(targetCase, movingCase, struct, displacementField,'Demons')
				except:
					print 'Case {0}, structure {1} not found - continuing'.format(movingCase.id, s)
					continue

			movingCase = None
			gc.collect()
