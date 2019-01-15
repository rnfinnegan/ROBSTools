import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import SimpleITK as sitk
import os
import numpy as np

from mpl_toolkits.axes_grid1 import make_axes_locatable

from ipywidgets import interact, interactive
from ipywidgets import widgets


def isotropicResample(inputImage, interpOrder = 2):
    """
    Input: SimpleITK image
    Output: Isotropically resampled image
    Notes:
      - isotropic spacing taken as smallest spacing in input image
      - linear interpolation used
    """
    sp = inputImage.GetSpacing()
    spacing = (min(sp),)*3

    si = inputImage.GetSize()
    size = tuple([int(sp[i]/spacing[0]*k+0.5) for i,k in enumerate(si)])

    r = sitk.ResampleImageFilter()
    #r.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    r.SetReferenceImage(inputImage)
    #r.SetOutputDirection(inputImage.GetDirection())
    #r.SetOutputOrigin(inputImage.GetOrigin())
    r.SetOutputSpacing(spacing)
    r.SetSize(size)
    r.SetInterpolator(interpOrder)

    isoImage = r.Execute(inputImage)
    return isoImage

def identicalResample(inputImage, referenceImage, interpOrder = 2):
    """
    Input: SimpleITK image
    Output: Resampled image (to same resolution as reference)
    Notes:
      - linear interpolation used
    """
    spacing = referenceImage.GetSpacing()
    size = referenceImage.GetSize()

    r = sitk.ResampleImageFilter()
    #r.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    #r.SetReferenceImage(referenceImage)
    #r.SetOutputDirection(referenceImage.GetDirection())
    #r.SetOutputOrigin(referenceImage.GetOrigin())
    r.SetOutputSpacing(spacing)
    r.SetSize(size)
    r.SetInterpolator(interpOrder)

    resampledImage = r.Execute(inputImage)
    return resampledImage

def ThresholdAndMeasureLungVolume(image, l=0, u=1):
    # Perform the threshold
    imThresh = sitk.Threshold(image, lower=l, upper=u)
    mask = sitk.ConnectedComponent(sitk.Cast(imThresh*1024, sitk.sitkInt32),fullyConnected=False)

    cts = np.bincount(sitk.GetArrayFromImage(mask).flatten())
    maxVals = cts.argsort()[-6:][::-1]

    PBR = np.zeros_like(maxVals, dtype=np.float32)
    label_shape_analysis = sitk.LabelShapeStatisticsImageFilter()
    for i, val in enumerate(maxVals):
        binaryVol = sitk.Equal(mask, val)
        testingVol = binaryVol#sitk.BinaryDilate(binaryVol)
        label_shape_analysis.Execute(testingVol)
        PBR[i] = label_shape_analysis.GetPerimeterOnBorderRatio(True)

    return PBR, mask, maxVals


def AutoLungSegment(image):
    voxSize = np.product(image.GetSpacing())/1000000. #conversion to litres
    imNorm = sitk.Normalize(sitk.Threshold(image, -1000,500, outsideValue=-1000))
    l = -0.3; u = 0.3
    #For debugging purposes
    #sitk.WriteImage(imNorm, "/media/HDD/Documents/University/PhD/Research/TEST.nii.gz")
    PBR, mask, labels = ThresholdAndMeasureLungVolume(imNorm,l,u)
    indices = np.array(np.where(PBR<=5e-4))
    print PBR

    if indices.size==0:
        print "     Warning - non-zero PBR"
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


def MaximalBoundingBox(structList, expansion=[0,0,0]):
    imCombined = reduce(lambda x,y: x+y, structList)
    LSA = sitk.LabelShapeStatisticsImageFilter()
    LSA.Execute(imCombined)
    maskBoxRaw = LSA.GetBoundingBox(1)
    maskBoxExpand = np.array(maskBoxRaw)+np.outer([-1,2],expansion).flatten()
    return tuple(maskBoxExpand)


def AutoPatientSegment(image):

    imScale = sitk.RescaleIntensity(image, outputMinimum=0, outputMaximum=100)
    imThresh = sitk.Threshold(imScale, lower=15, upper=30)
    mask = sitk.ConnectedComponent(imThresh,fullyConnected=True)

    cts = np.bincount(sitk.GetArrayFromImage(mask).flatten())
    maxVals = cts.argsort()[-5:][::-1]


    PerimeterOnBorderRatio = np.zeros(len(maxVals))
    PhysicalSize = np.zeros_like(PerimeterOnBorderRatio)
    for i, val in enumerate(maxVals):

        label_shape_analysis = sitk.LabelShapeStatisticsImageFilter()
        label_shape_analysis.Execute(sitk.Equal(mask, val))
        PerimeterOnBorderRatio[i]=label_shape_analysis.GetPerimeterOnBorderRatio(True)
        PhysicalSize[i] = label_shape_analysis.GetPhysicalSize(True)

    PerimeterOnBorderRatio[PerimeterOnBorderRatio==0.]=999
    cond =  (PhysicalSize/(1e6)>1) & (PhysicalSize>PhysicalSize[PhysicalSize.argsort()[2]])
    PatientVal = maxVals[(PhysicalSize*cond).argsort()[-2]]
    maskBinary = sitk.Equal(mask, PatientVal)
    label_shape_analysis = sitk.LabelShapeStatisticsImageFilter()
    label_shape_analysis.Execute(maskBinary)
    maskBox = label_shape_analysis.GetBoundingBox(True)

    return maskBox, maskBinary

def CropImage(image, maskBox):
    imCrop = sitk.RegionOfInterest(image, size=maskBox[3:], index=maskBox[:3])
    return imCrop

def Display3DSlices(img, title=None, margin=0.05, dpi=80, figsize=(8,8), axis='z'):
	try:
		nda = sitk.GetArrayFromImage(img)
	except:
		nda = img

	dim = len(nda.shape)
	if dim==3:
		zsize = nda.shape[0]
		ysize = nda.shape[1]
		xsize = nda.shape[2]
	elif dim==2:
		zsize = nda.shape[0]
		ysize = nda.shape[1]


	if dim==3:
		if axis=='x' or axis=='X':
			def callback(x=None):
				fig = plt.figure(figsize=figsize, dpi=dpi)
				ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
				plt.set_cmap("gray")
				if title:
					plt.title(title)
				ax.imshow(nda[:,:,x],interpolation=None, origin='lower', aspect=2)
				ax.axis('off')
				plt.show()
			interact(callback, x=(0,xsize-1))
		elif axis=='y' or axis=='Y':
			def callback(y=None):
				fig = plt.figure(figsize=figsize, dpi=dpi)
				ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
				plt.set_cmap("gray")
				if title:
					plt.title(title)
				ax.imshow(nda[:,y,:],interpolation=None, origin='lower', aspect=2)
				ax.axis('off')
				plt.show()
			interact(callback, y=(0,ysize-1))
		elif axis=='z' or axis=='Z':
			def callback(z=None):
				fig = plt.figure(figsize=figsize, dpi=dpi)
				ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
				plt.set_cmap("gray")
				if title:
					plt.title(title)
				ax.imshow(nda[z,:,:],interpolation=None)
				ax.axis('off')
				plt.show()
			interact(callback, z=(0,zsize-1))
		else:
			raise(ValueError('Axis must be one of "x", "y" or "z" (case insensitive)'))

	if dim==2:
		if axis=='x' or axis=='X':
			def callback(x=None):
				fig = plt.figure(figsize=figsize, dpi=dpi)
				ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
				plt.set_cmap("gray")
				if title:
					plt.title(title)
				ax.imshow(nda[:,x],interpolation=None, origin='lower')
				ax.axis('off')
				plt.show()
			interact(callback, x=(0,xsize-1))
		elif axis=='y' or axis=='Y':
			def callback(y=None):
				fig = plt.figure(figsize=figsize, dpi=dpi)
				ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
				plt.set_cmap("gray")
				if title:
					plt.title(title)
				ax.imshow(nda[:,y],interpolation=None, origin='lower')
				ax.axis('off')
				plt.show()
			interact(callback, y=(0,ysize-1))
		else:
			raise(ValueError('Axis must be one of "x", "y" or "z" (case insensitive)'))
		return fig


def Display3DVectorSlices(img, vectorImage, title=None, axis='z', skipStep=4, dpi=100):

    if axis=='z' or axis=='Z':
        zsize = img.GetSize()[2]
        nda=sitk.GetArrayFromImage(img)
        vf = sitk.GetArrayFromImage(vectorImage)
        u = sitk.GetArrayFromImage(sitk.VectorIndexSelectionCast(vectorImage, 0)) # x-component = u
        v = sitk.GetArrayFromImage(sitk.VectorIndexSelectionCast(vectorImage, 1)) # y-component = v
        w = sitk.GetArrayFromImage(sitk.VectorIndexSelectionCast(vectorImage, 2)) # z-component = w

        # the numpy 3D scalar arrays are indexed (z,y,x)


        x, y = np.mgrid[0:vf.shape[2]:1,0:vf.shape[1]:1]

        skip = (slice(None, None, skipStep), slice(None, None, skipStep))
        skip3D = (slice(None, None, skipStep),slice(None, None, skipStep), slice(None, None, None))
        x, y, u, v, w = x[skip], y[skip], u.T[skip3D], v.T[skip3D], w.T[skip3D]

        def callback(z=None):

            fig = plt.figure(figsize=(10,10),dpi=dpi)

            if title:
                plt.title(title)

            ax = fig.add_subplot(1,1,1)
            im=ax.quiver(x,y,-1.0*u[:,:,z],v[:,:,z],-1.0*w[:,:,z], cmap=plt.cm.Spectral)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical', label='Perpendicular Deformation (+=up)')

            ax.imshow(nda[z,:,:],cmap=plt.cm.Greys_r)

            ax.axis('off')
            ax.grid()

            plt.show()

        interact(callback, z=(0,zsize-1))

    if axis=='y' or axis=='y':
        ysize = img.GetSize()[1]
        nda=sitk.GetArrayFromImage(img)
        vf = sitk.GetArrayFromImage(vectorImage)
        u = sitk.GetArrayFromImage(sitk.VectorIndexSelectionCast(vectorImage, 0)) # x-component = u
        v = sitk.GetArrayFromImage(sitk.VectorIndexSelectionCast(vectorImage, 1)) # y-component = v
        w = sitk.GetArrayFromImage(sitk.VectorIndexSelectionCast(vectorImage, 2)) # z-component = w

        # the numpy 3D scalar arrays are indexed (z,y,x)


        x, z = np.mgrid[0:vf.shape[2]:1,0:vf.shape[0]:1]

        skip = (slice(None, None, skipStep), slice(None, None, skipStep))
        skip3D = (slice(None, None, skipStep),slice(None, None, None), slice(None, None, skipStep))
        x, z, u, v, w = x[skip], z[skip], u.T[skip3D], v.T[skip3D], w.T[skip3D]

        def callback(y=None):

            fig = plt.figure(figsize=(10,10), dpi=dpi)


            ax = fig.add_subplot(1,1,1, aspect=2)
            im=ax.quiver(x,z,-1.0*u[:,y,:],-1.0*w[:,y,:],v[:,y,:], cmap=plt.cm.Spectral)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical', label='Perpendicular Deformation')

            ax.imshow(nda[:,y,:],cmap=plt.cm.Greys_r, origin='lower', aspect=2)

            ax.axis('off')
            ax.grid()

            plt.show()

        interact(callback, y=(0,ysize-1))

    if axis=='x' or axis=='X':
        xsize = img.GetSize()[0]
        nda=sitk.GetArrayFromImage(img)
        vf = sitk.GetArrayFromImage(vectorImage)
        u = sitk.GetArrayFromImage(sitk.VectorIndexSelectionCast(vectorImage, 0)) # x-component = u
        v = sitk.GetArrayFromImage(sitk.VectorIndexSelectionCast(vectorImage, 1)) # y-component = v
        w = sitk.GetArrayFromImage(sitk.VectorIndexSelectionCast(vectorImage, 2)) # z-component = w

        # the numpy 3D scalar arrays are indexed (z,y,x)


        y, z = np.mgrid[0:vf.shape[1]:1,0:vf.shape[0]:1]

        skip = (slice(None, None, skipStep), slice(None, None, skipStep))
        skip3D = (slice(None, None, None),slice(None, None, skipStep), slice(None, None, skipStep))
        y, z, u, v, w = y[skip], z[skip], u.T[skip3D], v.T[skip3D], w.T[skip3D]

        def callback(x=None):
            fig = plt.figure(figsize=(10,10), dpi=dpi)

            ax = fig.add_subplot(1,1,1)
            ax.imshow(nda[:,:,x],cmap=plt.cm.Greys_r, origin='lower', aspect=2)
            im=ax.quiver(y,z,-1.0*v[x,:,:],-1.0*w[x,:,:],-1.0*u[x,:,:], cmap=plt.cm.Spectral)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical', label='Perpendicular Deformation')

            ax.axis('off')
            ax.grid()

            plt.show()

        interact(callback, x=(0,xsize-1))

def Display3DSlicesWithOverlay(img, overlayDict, title=None, margin=0.05, dpi=80, figsize=(8,8), axis='z'):
	try:
		nda = sitk.GetArrayFromImage(img)
	except:
		nda = img

	zsize = nda.shape[0]
	ysize = nda.shape[1]
	xsize = nda.shape[2]

	plotDict={}
	for name in overlayDict:
		try:
			plotDict[name]=sitk.GetArrayFromImage(overlayDict[name])
		except:
			plotDict[name]=overlayDict[name]

	niceCols = plt.get_cmap('Vega20')

	fig=0
	if axis=='x' or axis=='X':
		def callback(x=None):
			fig = plt.figure(figsize=figsize, dpi=dpi)
			ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
			plt.set_cmap("gray")
			if title:
				plt.title(title)
			ax.imshow(nda[:,:,x],interpolation=None, origin='lower', aspect=2)
			colIndex=0
			for name in sorted(plotDict):
				try:
					cntr = ax.contour(plotDict[name][:,:,x], origin='lower',colors=[niceCols.colors[colIndex]], linewidths=0.5)
					colIndex+=1
					ax.plot([0,0],[0,0], c=cntr.get_cmap().colors[0], label=name, lw=6.)
				except:
					0
			ax.axis('off')
			ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=min([len(plotDict),4]), mode="expand", borderaxespad=0.)
			fig.show()
		interact(callback, x=(0,xsize-1))

	elif axis=='y' or axis=='Y':
		def callback(y=None):
			fig = plt.figure(figsize=figsize, dpi=dpi)
			ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
			plt.set_cmap("gray")
			if title:
				plt.title(title)
			ax.imshow(nda[:,y,:],interpolation=None, origin='lower', aspect=2)
			colIndex=0
			for name in sorted(plotDict):
				try:
					cntr = ax.contour(plotDict[name][:,y,:], origin='lower',colors=[niceCols.colors[colIndex]], linewidths=0.5)
					colIndex+=1
					ax.plot([0,0],[0,0], c=cntr.get_cmap().colors[0], label=name, lw=6.)
				except:
					0
			ax.axis('off')
			ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=min([len(plotDict),4]), mode="expand", borderaxespad=0.)
			fig.show()
		interact(callback, y=(0,ysize-1))

	elif axis=='z' or axis=='Z':
		def callback(z=None):
			fig = plt.figure(figsize=figsize, dpi=dpi)
			ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
			plt.set_cmap("gray")
			if title:
				plt.title(title)
			ax.imshow(nda[z,:,:],interpolation=None)
			colIndex=0
			for name in sorted(plotDict):
				try:
					cntr = ax.contour(plotDict[name][z,:,:],colors=[niceCols.colors[colIndex]], linewidths=0.5)
					colIndex+=1
					ax.plot([0,0],[0,0], c=cntr.get_cmap().colors[0], label=name, lw=6.)
				except:
					0
			ax.axis('off')
			ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=min([len(plotDict),4]), mode="expand", borderaxespad=0.)
			fig.show()
		interact(callback, z=(0,zsize-1))

	else:
		raise(ValueError('Axis must be one of "x", "y" or "z" (case insensitive)'))

	return fig
