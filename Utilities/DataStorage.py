# import itk
import SimpleITK as sitk
from Config import filelocs

"""
To do:
- create function to convert between itk and simpleitk?
- split into target and fixed cases?
"""

class Case:
	"""
	A case is a class that holds the following attributes:
		image: the 3D image (with associated attruibutes)
		structures: the 3D labels (segmentations), which are themselves images
	Note: structures is a dict {name:image}
	"""

	def __init__(self, id, image=None, imageLoc=False, regType='Original'):
		self.id = id
		#reader = itk.ImageFileReader.IF3.New(FileName=imageloc)
		#reader.Update()
		#image = reader.GetOutput()

		if imageLoc:
			#print 'Image location used'
			image = sitk.ReadImage(imageLoc)

		self.image = image
		self.regType = regType
		self.structures = {}

	def addStructure(self, name, image=None, imageLoc=False):
		#reader = itk.ImageFileReader.IF3.New(FileName=imageloc)
		#reader.Update()
		#image = reader.GetOutput()
		if imageLoc:
			#print 'Image location used'
			image = sitk.ReadImage(imageLoc)
		self.structures[name] = image

	def delStructure(self, name):
		self.structures.pop(name)

	def addRegStruct(self, targetId, name):
		try:
			self.addStructure(name=name, imageLoc='{0}/Case_{1}_to_Case_{2}_{3}_{4}.nii.gz'.format(filelocs.regStructLocation(targetId, self.regType), self.id, targetId, name, regTypeFormatter(self.regType)))
		except:
			print 'Structure {0} not found for Case {1}'.format(name, self.id)

	def addStructsFromList(self, structList, directory, basename, suffix):
		for s in structList:
			try:
				#print "{0}/{1}{2}{3}.nii.gz".format(directory, basename, s, suffix)
				self.addStructure(name=s, imageLoc="{0}/{1}{2}{3}.nii.gz".format(directory, basename, s, suffix))
			except:
				0

def importFixedCase(id, structList=[]):
	fCase = Case(id=id, imageLoc='{0}/Case_{1}_crop.nii.gz'.format(filelocs.fixedImageLocation, id))
	for s in structList:
		try:
			fCase.addStructure(name=s, imageLoc='{0}/Case_{1}_{2}_crop.nii.gz'.format(filelocs.fixedStructLocation, id, s))
		except:
			continue
	return fCase

def importRegCase(Id, targetId, regType, structList):
	rCase = Case(id=Id, imageLoc='{0}/Case_{1}_to_Case_{2}_{3}.nii.gz'.format(filelocs.regImageLocation(targetId, regType), Id, targetId, regTypeFormatter(regType)), regType=regType)
	for s in structList:
		try:
			rCase.addStructure(name=s, imageLoc='{0}/Case_{1}_to_Case_{2}_{3}_{4}.nii.gz'.format(filelocs.regStructLocation(targetId, regType), Id, targetId, s, regTypeFormatter(regType)))
		except:
			continue
	return rCase




def regTypeFormatter(regType):
	if regType.lower()=='bsplines':
		return 'nrr'
	elif regType.lower()=='demons':
		return 'nrr'
	elif regType.lower()=='rigid':
		return 'rigid'
	else:
		raise ValueError('Must use valid registration type.')
